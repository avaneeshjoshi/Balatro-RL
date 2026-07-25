"""CLI for preparing planner examples for supervised training."""

import argparse
import json
from pathlib import Path

from ai_agent.training_data import prepare_training_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and aggregate planner examples into train/validation data",
    )
    parser.add_argument(
        "--data",
        default="data/balatrobench/planner_examples.jsonl",
        help="Audited planner examples JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/balatrobench/training",
        help="Output directory",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of unique visible states reserved for validation",
    )
    args = parser.parse_args()
    manifest = prepare_training_data(
        planner_path=Path(args.data),
        output_dir=Path(args.output),
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["splits"], indent=2))
    print(f"Manifest: {Path(args.output).resolve() / 'planner_training_manifest.json'}")


if __name__ == "__main__":
    main()
