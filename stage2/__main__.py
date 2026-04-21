"""``python -m stage2 list|smoke|run ...`` — run from repository root."""

from __future__ import annotations

import argparse
import os
import sys


def _find_entry(task: str, name: str):
    from stage2.registry import all_stage2_datasets

    for d in all_stage2_datasets():
        if d.task == task and d.path_segment == name:
            return d
    return None


def _cmd_list() -> int:
    from stage2.registry import all_stage2_datasets

    for d in all_stage2_datasets():
        print(f"{d.task:22}  {d.path_segment:20}  {d.dataset_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m stage2")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Print registered datasets")

    p_smoke = sub.add_parser("smoke", help="All datasets, light CONFIG + 1 LLM call per round")
    p_smoke.add_argument(
        "--only",
        nargs=2,
        metavar=("TASK", "DATASET_DIR"),
        action="append",
        help="Repeatable, e.g. --only regression boston",
    )

    p_run = sub.add_parser("run", help="Paper-style CONFIG for one dataset")
    p_run.add_argument("task")
    p_run.add_argument("dataset_dir")

    args = p.parse_args(argv)

    if args.cmd == "list":
        return _cmd_list()

    if args.cmd == "smoke":
        from stage2.defaults import smoke_config
        from stage2.registry import all_stage2_datasets
        from stage2.runner import run_stage2_for_dataset

        n_rounds = int(os.environ.get("OURS_N_ROUNDS", "1"))
        cfg = smoke_config()
        if args.only:
            entries = []
            for t, d in args.only:
                found = _find_entry(t, d)
                if not found:
                    print(f"Unknown: {t!r} {d!r}", file=sys.stderr)
                    return 2
                entries.append(found)
        else:
            entries = all_stage2_datasets()
        fails = 0
        for e in entries:
            print("\n" + "=" * 60)
            ok, msg = run_stage2_for_dataset(
                e,
                n_rounds=n_rounds,
                config=cfg,
                output_subdir=os.environ.get("OURS_STAGE2_OUTPUT_DIR", "outputs_stage2_refactor"),
                log_subdir=os.environ.get("OURS_STAGE2_LOG_DIR", "logs_stage2_refactor"),
            )
            if not ok:
                print(f"[FAIL] {e.task}/{e.path_segment}: {msg}")
                fails += 1
        print("\n" + "=" * 60)
        print(f"Smoke finished. failures={fails} / {len(entries)}")
        return 1 if fails else 0

    if args.cmd == "run":
        from stage2.defaults import paper_config
        from stage2.runner import run_stage2_for_dataset

        entry = _find_entry(args.task, args.dataset_dir)
        if entry is None:
            print("Unknown task/dataset. Try: python -m stage2 list", file=sys.stderr)
            return 2
        n_rounds = int(os.environ.get("OURS_N_ROUNDS", "5"))
        ok, msg = run_stage2_for_dataset(
            entry,
            n_rounds=n_rounds,
            config=paper_config(),
            output_subdir=os.environ.get("OURS_STAGE2_OUTPUT_DIR", "outputs_stage2"),
            log_subdir=os.environ.get("OURS_STAGE2_LOG_DIR", "logs_stage2"),
        )
        if not ok:
            print(msg, file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
