"""CLI for normalizing a local BalatroBench dataset."""

import argparse
import json
from pathlib import Path

from ai_agent.balatrobench_importer import import_balatrobench


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BalatroBench runs into canonical JSONL tables",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing BalatroBench run folders",
    )
    parser.add_argument(
        "--output",
        default="data/balatrobench",
        help="Output directory (default: data/balatrobench)",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Copy tool-call reasoning into transitions (substantially larger output)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Import only the first N runs for a smoke test",
    )
    args = parser.parse_args()

    manifest = import_balatrobench(
        source_root=Path(args.source),
        output_dir=Path(args.output),
        include_reasoning=args.include_reasoning,
        max_runs=args.max_runs,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {Path(args.output).resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
