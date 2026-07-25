"""CLI for auditing generated planner examples before training."""

import argparse
import json
from pathlib import Path

from ai_agent.planner_dataset import audit_planner_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit planner JSONL for privacy, legality, and join consistency",
    )
    parser.add_argument(
        "--data",
        default="data/balatrobench/planner_examples.jsonl",
        help="Planner JSONL path",
    )
    parser.add_argument(
        "--report",
        help="Audit JSON path (default: next to data as planner_audit.json)",
    )
    args = parser.parse_args()
    audit = audit_planner_dataset(
        dataset_path=Path(args.data),
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps({"status": audit["status"], **audit["counts"]}, indent=2))
    if audit["status"] != "passed":
        print(json.dumps(audit["issues"], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
