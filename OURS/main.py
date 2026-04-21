#!/usr/bin/env python
# coding: utf-8
"""
Boston regression Stage-2 — wrapper around ``stage2.runner``.

From repo root::

    python OURS/main.py

Requires ``OPENAI_API_KEY`` (optional ``OPENAI_BASE_URL``).
``OURS_FAST_SEMANTIC=1`` disables pairwise semantic-graph LLM (lower cost).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from openai import OpenAI

    from stage2.defaults import paper_config
    from stage2.registry import Stage2Dataset
    from stage2.runner import run_stage2_for_dataset

    entry = Stage2Dataset("regression", "boston", "boston_augmented")
    n_rounds = int(os.environ.get("OURS_N_ROUNDS", "5"))
    cfg = dict(paper_config())
    if os.environ.get("OURS_FAST_SEMANTIC", "").lower() in ("1", "true", "yes"):
        cfg["use_llm_semantic_graph"] = False

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.laozhang.ai/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    np.random.seed(int(os.environ.get("OURS_RANDOM_SEED", "42")))

    ok, msg = run_stage2_for_dataset(
        entry,
        n_rounds=n_rounds,
        config=cfg,
        output_subdir=os.environ.get("OURS_STAGE2_OUTPUT_DIR", "outputs_stage2_ours"),
        log_subdir=os.environ.get("OURS_STAGE2_LOG_DIR", "logs_stage2_ours"),
        llm_client=client,
    )
    if not ok:
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
