"""Validate exact-eligible base scores against BalatroBench transitions."""

import argparse
import json
from pathlib import Path

from ai_agent.scoring_engine import HiddenCardError, ScoringError, score_play


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base engine scores to observed scores")
    parser.add_argument(
        "--data",
        default="data/balatrobench/planner_examples.jsonl",
        help="Planner examples JSONL",
    )
    parser.add_argument(
        "--report",
        default="data/balatrobench/scoring_validation.json",
        help="Validation report JSON",
    )
    args = parser.parse_args()
    counts = {
        "examples": 0,
        "structured_plays": 0,
        "exact_eligible": 0,
        "matched": 0,
        "mismatched": 0,
        "skipped_hidden_cards": 0,
        "scoring_errors": 0,
    }
    mismatch_samples = []
    error_samples = []
    with Path(args.data).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            example = json.loads(line)
            counts["examples"] += 1
            action = example.get("action", {})
            quality = example.get("quality", {})
            observed = example.get("outcome", {}).get("score_delta")
            if (
                action.get("kind") != "play"
                or quality.get("source_tier") != "structured_validated"
                or not isinstance(observed, (int, float))
            ):
                continue
            counts["structured_plays"] += 1
            try:
                result = score_play(example["state"], action["card_indices"])
            except HiddenCardError:
                counts["skipped_hidden_cards"] += 1
                continue
            except ScoringError as exc:
                counts["scoring_errors"] += 1
                if len(error_samples) < 10:
                    error_samples.append({"example_id": example.get("example_id"), "error": str(exc)})
                continue
            if not result.exact:
                continue
            counts["exact_eligible"] += 1
            if result.score == observed:
                counts["matched"] += 1
            else:
                counts["mismatched"] += 1
                if len(mismatch_samples) < 10:
                    mismatch_samples.append(
                        {
                            "example_id": example.get("example_id"),
                            "hand_type": result.hand_type,
                            "predicted": result.score,
                            "observed": observed,
                            "cards": action.get("card_indices"),
                        }
                    )
    report = {
        "status": "passed" if counts["mismatched"] == 0 and counts["scoring_errors"] == 0 else "failed",
        "counts": counts,
        "match_rate": counts["matched"] / counts["exact_eligible"] if counts["exact_eligible"] else None,
        "mismatch_samples": mismatch_samples,
        "error_samples": error_samples,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
