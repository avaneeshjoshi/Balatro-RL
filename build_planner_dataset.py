"""CLI for building planner-ready BalatroBench examples."""

import argparse
import json
from pathlib import Path

from ai_agent.planner_dataset import build_planner_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse normalized BalatroBench decisions into planner examples",
    )
    parser.add_argument(
        "--input",
        default="data/balatrobench",
        help="Normalized BalatroBench directory (default: data/balatrobench)",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: INPUT/planner_examples.jsonl)",
    )
    parser.add_argument(
        "--manifest",
        help="Validation manifest path (default: INPUT/planner_manifest.json)",
    )
    args = parser.parse_args()

    manifest = build_planner_dataset(
        input_dir=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Examples: {manifest['output']}")
    print(f"Validation: {(Path(args.manifest).resolve() if args.manifest else Path(args.input).resolve() / 'planner_manifest.json')}")


if __name__ == "__main__":
    main()
