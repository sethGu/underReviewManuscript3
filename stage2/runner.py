"""Load data, call ``FS_method_pipeline``, save round CSVs + JSON log."""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from stage2.defaults import OPERATOR_BASE, TEMPLATE_BASE
from stage2.paths import ours_package_dir
from stage2.registry import Stage2Dataset, dataset_base_dir


def _ensure_fs_method_importable() -> None:
    """Insert ``OURS/`` on ``sys.path`` and load ``FS_method`` once.

    Some Anaconda stacks ship a broken ``dask``/``pandas`` combo: ``import lightgbm``
    then pulls ``dask.dataframe`` and raises ``AttributeError``, which LightGBM does
    not catch. Forcing ``dask`` import to fail makes LightGBM skip optional Dask
    support (fine for this project).
    """
    ours = ours_package_dir()
    if not (ours / "FS_method.py").exists():
        raise FileNotFoundError(f"Missing FS_method.py under {ours}")
    s = str(ours)
    if s not in sys.path:
        sys.path.insert(0, s)
    if "FS_method" in sys.modules:
        return

    import builtins

    _real_import = builtins.__import__

    def _import_guard(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "dask" or name.startswith("dask.")):
            raise ImportError("dask disabled for LightGBM import compatibility")
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _import_guard
    try:
        import FS_method  # noqa: F401 — loads lightgbm + pipeline
    finally:
        builtins.__import__ = _real_import


def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    path = os.path.normpath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def apply_features(df: pd.DataFrame, feature_formulas: dict) -> pd.DataFrame:
    df_new = df.copy()
    for feat_name, info in feature_formulas.items():
        cols = info["columns"]
        template = info["template"]
        expr = template
        mapping = {}
        if len(cols) >= 1:
            mapping["x"] = cols[0]
        if len(cols) >= 2:
            mapping["y"] = cols[1]
        if len(cols) >= 3:
            mapping["z"] = cols[2]
        for sym, col in mapping.items():
            expr = re.sub(rf"\b{sym}\b", f'df_new["{col}"]', expr)
        try:
            df_new[feat_name] = eval(expr, {"df_new": df_new, "np": np})
        except Exception as e:
            print(f"[WARN] Failed to materialize {feat_name}: {e}")
    return df_new


def default_llm_client() -> OpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.laozhang.ai/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) for Stage-2 with LLM."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def run_stage2_for_dataset(
    entry: Stage2Dataset,
    *,
    n_rounds: int,
    config: Dict[str, Any],
    output_subdir: str = "outputs_stage2",
    log_subdir: str = "logs_stage2",
    llm_client: Optional[OpenAI] = None,
    random_seed: int = 42,
) -> Tuple[bool, str]:
    _ensure_fs_method_importable()
    from FS_method import FS_method_pipeline, FS_method_result

    base = dataset_base_dir(entry)
    if not base.is_dir():
        return False, f"missing directory {base}"

    dataset = entry.dataset_id
    train_path = base / f"{dataset}_train.csv"
    test_path = base / f"{dataset}_test.csv"
    if not train_path.is_file():
        return False, f"missing {train_path}"
    if not test_path.is_file():
        return False, f"missing {test_path}"

    task_path = base / "origin_data_task_description.txt"
    feat_path = base / "CAAFE_feature_description.txt"
    if not task_path.is_file():
        return False, f"missing {task_path}"
    if not feat_path.is_file():
        return False, f"missing {feat_path}"

    out_dir = base / output_subdir
    log_dir = base / log_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    target_col = df_train.columns[-1]

    task_description = task_path.read_text(encoding="utf-8")
    feat_description = feat_path.read_text(encoding="utf-8")

    client = llm_client if llm_client is not None else default_llm_client()
    np.random.seed(random_seed)

    all_round_logs: List[dict] = []
    current_train = df_train.copy()
    current_test = df_test.copy()
    current_feat_description = feat_description
    total_feature_gen_time = 0.0

    print(f"[INFO] {entry.task}/{entry.path_segment}  target={target_col!r}  rounds={n_rounds}")
    print(f"       train={df_train.shape} test={df_test.shape}")

    for round_id in range(1, n_rounds + 1):
        print(f"\n========== AutoFE Round {round_id}  ({entry.task}/{entry.path_segment}) ==========")
        t0 = time.perf_counter()
        w0 = datetime.datetime.now()

        result: FS_method_result = FS_method_pipeline(
            df=current_train,
            target_col=target_col,
            task_type=entry.task,
            config=config,
            operator_base=OPERATOR_BASE,
            template_base=TEMPLATE_BASE,
            llm_client=client,
            task_desc=task_description,
            feat_desc=current_feat_description,
        )
        elapsed = time.perf_counter() - t0
        w1 = datetime.datetime.now()
        total_feature_gen_time += elapsed

        print(f"[INFO] Round {round_id} selected: {result.selected_features}")

        current_train = apply_features(current_train, result.feature_formulas)
        current_test = apply_features(current_test, result.feature_formulas)

        train_out = out_dir / f"{dataset}_round{round_id}_train.csv"
        test_out = out_dir / f"{dataset}_round{round_id}_test.csv"
        _atomic_to_csv(current_train, str(train_out))
        _atomic_to_csv(current_test, str(test_out))
        print(f"[INFO] Saved {train_out.name} , {test_out.name}")

        all_round_logs.append(
            {
                "round": round_id,
                "feature_gen_start_time": str(w0),
                "feature_gen_end_time": str(w1),
                "feature_gen_elapsed_seconds": elapsed,
                "selected_features": result.selected_features,
                "feature_formulas": result.feature_formulas,
                "operator_whitelist": result.operator_whitelist,
                "template_whitelist": result.template_whitelist,
                "task_description": task_description,
                "feat_description": current_feat_description,
                "debug_info": result.debug_info,
            }
        )

        if result.feature_formulas:
            added = "\n".join(
                f"- {name}: derived from {info['columns']} using `{info['template']}`"
                for name, info in result.feature_formulas.items()
            )
            current_feat_description = (
                current_feat_description
                + f"\n\n[AutoFE Round {round_id} Added Features]\n"
                + added
            )

    log_path = log_dir / f"{dataset}_autofe_full_log.json"
    log_path.write_text(json.dumps(all_round_logs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[TIME] Total feature generation {total_feature_gen_time:.2f}s")
    print(f"[INFO] Log -> {log_path}")
    return True, "ok"
