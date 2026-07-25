"""Prepare audited planner examples for supervised candidate-policy training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            yield value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def state_fingerprint(state: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(state).encode("ascii")).hexdigest()


def _action_key(kind: str, card_indices: list[int]) -> str:
    # Balatro scores selected cards in their hand order. The tool's click order
    # does not change the resulting selected set, so masks are canonicalized.
    return _canonical_json([kind, sorted(card_indices)])


def _split_for_fingerprint(fingerprint: str, validation_fraction: float) -> str:
    position = int(fingerprint[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return "validation" if position < validation_fraction else "train"


def _write_jsonl(output_file: Any, record: dict[str, Any]) -> None:
    output_file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def prepare_training_data(
    *,
    planner_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Filter, deduplicate, aggregate, split, and audit planner examples."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    source = Path(planner_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    for example in _iter_jsonl(source):
        counts["input_examples"] += 1
        quality = example.get("quality")
        if not isinstance(quality, dict) or not quality.get("usable"):
            counts["rejected_examples"] += 1
            reasons = []
            if isinstance(quality, dict):
                reasons = [*quality.get("parse_errors", []), *quality.get("action_errors", [])]
            for reason in reasons or ["quality.usable is false or missing"]:
                rejected_reasons[str(reason)] += 1
            continue
        state = example.get("state")
        action = example.get("action")
        provenance = example.get("provenance")
        outcome = example.get("outcome")
        if not isinstance(state, dict) or not isinstance(action, dict):
            counts["rejected_examples"] += 1
            rejected_reasons["missing structured state or action"] += 1
            continue
        if _contains_key(state, "seed"):
            raise ValueError(f"Seed leaked into policy state for {example.get('example_id')}")
        kind = action.get("kind")
        card_indices = action.get("card_indices")
        if kind not in {"play", "discard"} or not isinstance(card_indices, list):
            counts["rejected_examples"] += 1
            rejected_reasons["invalid action shape"] += 1
            continue

        fingerprint = state_fingerprint(state)
        group = groups.setdefault(
            fingerprint,
            {
                "state": state,
                "actions": {},
                "source_examples": 0,
                "runs": set(),
                "models": set(),
                "seeds": set(),
                "winning_run_examples": 0,
            },
        )
        if group["state"] != state:
            raise RuntimeError(f"State fingerprint collision: {fingerprint}")
        model = str(provenance.get("model") or "unknown") if isinstance(provenance, dict) else "unknown"
        run_id = str(example.get("run_id") or "unknown")
        seed = provenance.get("seed") if isinstance(provenance, dict) else None
        action_key = _action_key(str(kind), card_indices)
        action_group = group["actions"].setdefault(
            action_key,
            {
                "kind": kind,
                "card_indices": sorted(card_indices),
                "raw_votes": 0,
                "models": set(),
                "runs": set(),
                "model_counts": Counter(),
                "winning_run_votes": 0,
                "structured_votes": 0,
            },
        )
        action_group["raw_votes"] += 1
        action_group["models"].add(model)
        action_group["runs"].add(run_id)
        action_group["model_counts"][model] += 1
        action_group["winning_run_votes"] += int(
            isinstance(outcome, dict) and bool(outcome.get("run_won"))
        )
        action_group["structured_votes"] += int(quality.get("source_tier") == "structured_validated")
        group["source_examples"] += 1
        group["runs"].add(run_id)
        group["models"].add(model)
        if seed is not None:
            group["seeds"].add(str(seed))
        group["winning_run_examples"] += int(
            isinstance(outcome, dict) and bool(outcome.get("run_won"))
        )
        counts["accepted_examples"] += 1

    output_paths = {
        "train": destination / "planner_train.jsonl",
        "validation": destination / "planner_validation.jsonl",
    }
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in output_paths.items()
    }
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: Counter[str] = Counter()
    consensus_buckets: Counter[str] = Counter()
    integrity_errors: list[str] = []
    outputs: dict[str, Any] = {}
    try:
        outputs = {
            name: path.open("w", encoding="utf-8", newline="\n")
            for name, path in temporary_paths.items()
        }
        for fingerprint, group in sorted(groups.items()):
            split = _split_for_fingerprint(fingerprint, validation_fraction)
            model_action_totals: dict[str, int] = Counter()
            for action_group in group["actions"].values():
                for model, model_count in action_group["model_counts"].items():
                    model_action_totals[model] += model_count

            actions: list[dict[str, Any]] = []
            for action_group in group["actions"].values():
                model_vote = sum(
                    count / model_action_totals[model]
                    for model, count in action_group["model_counts"].items()
                )
                probability = model_vote / len(group["models"])
                action_record = {
                    "kind": action_group["kind"],
                    "card_indices": action_group["card_indices"],
                    "target_probability": probability,
                    "model_vote": model_vote,
                    "raw_votes": action_group["raw_votes"],
                    "source_models": len(action_group["models"]),
                    "source_runs": len(action_group["runs"]),
                    "winning_run_votes": action_group["winning_run_votes"],
                    "structured_votes": action_group["structured_votes"],
                }
                actions.append(action_record)
                action_counts[str(action_group["kind"])] += action_group["raw_votes"]
            actions.sort(
                key=lambda item: (
                    -item["target_probability"],
                    -item["raw_votes"],
                    item["kind"],
                    item["card_indices"],
                )
            )
            probability_sum = sum(action["target_probability"] for action in actions)
            if abs(probability_sum - 1.0) > 1e-9:
                integrity_errors.append(
                    f"{fingerprint}: action probabilities sum to {probability_sum}"
                )
            consensus_probability = actions[0]["target_probability"]
            if consensus_probability == 1.0:
                bucket = "1.00"
            elif consensus_probability >= 0.75:
                bucket = "0.75-0.99"
            elif consensus_probability >= 0.5:
                bucket = "0.50-0.74"
            else:
                bucket = "below-0.50"
            consensus_buckets[bucket] += 1

            record = {
                "schema_version": SCHEMA_VERSION,
                "state_id": f"visible-state-{fingerprint}",
                "split": split,
                "state": group["state"],
                "action_targets": actions,
                "consensus_action": {
                    "kind": actions[0]["kind"],
                    "card_indices": actions[0]["card_indices"],
                    "target_probability": consensus_probability,
                },
                "provenance_summary": {
                    "source_examples": group["source_examples"],
                    "source_runs": len(group["runs"]),
                    "source_models": len(group["models"]),
                    "source_seeds": sorted(group["seeds"]),
                    "winning_run_examples": group["winning_run_examples"],
                },
                "quality": {
                    "teacher_quality": "aggregated_unrated_llm_votes",
                    "has_action_conflict": len(actions) > 1,
                    "action_options": len(actions),
                },
            }
            _write_jsonl(outputs[split], record)
            split_counts[split]["states"] += 1
            split_counts[split]["source_examples"] += group["source_examples"]
            split_counts[split]["conflicting_states"] += int(len(actions) > 1)
        for output_file in outputs.values():
            output_file.close()
        outputs = {}
        if integrity_errors:
            raise RuntimeError("; ".join(integrity_errors[:5]))
        for name, temporary_path in temporary_paths.items():
            temporary_path.replace(output_paths[name])
    except BaseException:
        for output_file in outputs.values():
            output_file.close()
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise

    counts["unique_visible_states"] = len(groups)
    counts["duplicate_examples_merged"] = counts["accepted_examples"] - len(groups)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output_dir": str(destination),
        "validation_fraction": validation_fraction,
        "split_strategy": "deterministic_visible_state_fingerprint",
        "status": "passed",
        "counts": dict(sorted(counts.items())),
        "splits": {
            name: dict(sorted(values.items())) for name, values in sorted(split_counts.items())
        },
        "actions": dict(sorted(action_counts.items())),
        "consensus_probability_buckets": dict(sorted(consensus_buckets.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "files": {name: path.name for name, path in output_paths.items()},
        "notes": [
            "Seeds are provenance only and are not part of the policy state or split key.",
            "Validation monitors visible-state generalization; final evaluation must use new live-game seeds.",
            "Targets are equal-weight model votes, not oracle labels or action values.",
        ],
    }
    manifest_path = destination / "planner_training_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")
    return manifest
