import json
import tempfile
import unittest
from pathlib import Path

from ai_agent.training_data import prepare_training_data, state_fingerprint


def example(
    example_id: str,
    *,
    state: dict,
    kind: str,
    cards: list[int],
    model: str,
    run_id: str,
    usable: bool = True,
) -> dict:
    return {
        "example_id": example_id,
        "run_id": run_id,
        "state": state,
        "action": {"kind": kind, "card_indices": cards, "selected_cards": []},
        "outcome": {"run_won": False},
        "provenance": {"model": model, "seed": "AAAAAAA"},
        "quality": {
            "usable": usable,
            "source_tier": "rendered_only",
            "parse_errors": [],
            "action_errors": [] if usable else ["bad action"],
        },
    }


class TrainingDataTests(unittest.TestCase):
    def test_state_fingerprint_ignores_dictionary_order(self):
        self.assertEqual(state_fingerprint({"a": 1, "b": 2}), state_fingerprint({"b": 2, "a": 1}))

    def test_filters_aggregates_and_equal_weights_models(self):
        state = {
            "phase": "SELECTING_HAND",
            "hand": {"count": 2, "cards": [{"index": 0}, {"index": 1}]},
        }
        rows = [
            example("a", state=state, kind="play", cards=[1, 0], model="model-a", run_id="run-a"),
            example("b", state=state, kind="play", cards=[0, 1], model="model-a", run_id="run-b"),
            example("c", state=state, kind="discard", cards=[1], model="model-b", run_id="run-c"),
            example(
                "d",
                state=state,
                kind="play",
                cards=[9],
                model="model-c",
                run_id="run-d",
                usable=False,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "planner.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            manifest = prepare_training_data(
                planner_path=source,
                output_dir=root / "output",
                validation_fraction=0.5,
            )
            records = []
            for name in ("planner_train.jsonl", "planner_validation.jsonl"):
                path = root / "output" / name
                records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())

            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["counts"]["accepted_examples"], 3)
            self.assertEqual(manifest["counts"]["rejected_examples"], 1)
            self.assertEqual(manifest["counts"]["unique_visible_states"], 1)
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["quality"]["has_action_conflict"])
            targets = {
                (item["kind"], tuple(item["card_indices"])): item["target_probability"]
                for item in records[0]["action_targets"]
            }
            self.assertEqual(targets[("play", (0, 1))], 0.5)
            self.assertEqual(targets[("discard", (1,))], 0.5)
            self.assertEqual(records[0]["provenance_summary"]["source_seeds"], ["AAAAAAA"])
            self.assertNotIn("seed", records[0]["state"])


if __name__ == "__main__":
    unittest.main()
