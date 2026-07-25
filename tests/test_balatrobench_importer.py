import json
import tempfile
import unittest
from pathlib import Path

from ai_agent.balatrobench_importer import import_balatrobench, sanitize_visible_state


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def response(turn: int, name: str, arguments: dict) -> dict:
    return {
        "custom_id": f"request-{turn:05d}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        },
        "error": None,
    }


def state(phase: str, chips: int, hands_left: int) -> dict:
    return {
        "seed": "AAAAAAA",
        "won": False,
        "state": phase,
        "round_num": 1,
        "ante_num": 1,
        "round": {
            "chips": chips,
            "hands_left": hands_left,
            "discards_left": 3,
        },
        "hand": {
            "count": 2,
            "cards": [
                {
                    "key": "D_A",
                    "value": {"rank": "A", "suit": "D"},
                    "state": {"hidden": True},
                },
                {"key": "C_2", "value": {"rank": "2", "suit": "C"}},
            ],
        },
        "cards": {
            "count": 2,
            "cards": [
                {"key": "S_K", "id": 2, "state": {"hidden": True}},
                {"key": "C_3", "id": 1, "state": {"hidden": True}},
            ],
        },
        "jokers": {"count": 0, "cards": []},
        "consumables": {"count": 0, "cards": []},
        "blinds": {},
        "hands": {},
    }


def request(turn: int, phase: str = "SELECTING_HAND") -> dict:
    return {
        "custom_id": f"request-{turn:05d}",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "guide"},
                        {
                            "type": "text",
                            "text": (
                                "# Current Game State\n\n"
                                f"- **Phase**: Playing Phase (gamestate is {phase})\n"
                                "- **Seed**: AAAAAAA\n\n"
                                "## Current Hand\n\n- 0: A of D (`D_A`)\n\n"
                                "# Tools available\nignored"
                            ),
                        },
                    ],
                }
            ]
        },
    }


class BalatroBenchImporterTests(unittest.TestCase):
    def test_imports_normalized_tables_and_marks_clean_in_blind_transition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runs"
            run_dir = source / "v1.0.8" / "default" / "vendor" / "model" / "run-1"
            output = root / "output"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "task.json",
                {
                    "model": {"vendor": "vendor", "name": "model"},
                    "seed": "AAAAAAA",
                    "deck": "RED",
                    "stake": "WHITE",
                },
            )
            write_json(
                run_dir / "stats.json",
                {"run_won": False, "final_ante": 1, "final_round": 1},
            )
            write_json(run_dir / "strategy.json", {"name": "Default"})
            with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(request(1, "SHOP")) + "\n")
                output_file.write(json.dumps(request(2)) + "\n")
            with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as output_file:
                output_file.write(
                    json.dumps(response(1, "next_round", {"reasoning": "advance"})) + "\n"
                )
                output_file.write(
                    json.dumps(response(2, "play", {"cards": [0], "reasoning": "best"})) + "\n"
                )
            with (run_dir / "gamestates.jsonl").open("w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(state("SELECTING_HAND", 0, 1)) + "\n")
                output_file.write(json.dumps(state("GAME_OVER", 10, 0)) + "\n")

            manifest = import_balatrobench(source_root=source, output_dir=output)

            self.assertEqual(manifest["counts"]["runs"], 1)
            self.assertEqual(manifest["counts"]["states"], 2)
            self.assertEqual(manifest["counts"]["transitions"], 2)
            self.assertEqual(manifest["counts"]["bc_candidates"], 1)
            self.assertEqual(manifest["counts"]["text_bc_candidates"], 1)

            states = [
                json.loads(line)
                for line in (output / "states.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            visible = states[0]["visible_state"]
            self.assertNotIn("seed", visible)
            self.assertNotIn("won", visible)
            self.assertEqual(visible["hand"]["cards"][0], {"hidden": True, "state": {"hidden": True}})
            self.assertEqual(
                [card["key"] for card in visible["cards"]["cards"]],
                ["C_3", "S_K"],
            )
            transitions = [
                json.loads(line)
                for line in (output / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIsNone(transitions[0]["pre_state_id"])
            self.assertTrue(transitions[1]["bc_candidate"])
            self.assertEqual(transitions[1]["action"]["arguments"], {"cards": [0]})
            self.assertEqual(transitions[1]["score_delta"], 10.0)
            self.assertEqual(transitions[1]["round_result"], "lost")
            self.assertEqual(
                transitions[1]["pre_state_id"],
                states[0]["state_id"],
            )
            request_states = [
                json.loads(line)
                for line in (output / "request_states.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(request_states), 2)
            self.assertNotIn("AAAAAAA", request_states[1]["state_text"])
            self.assertNotIn("Tools available", request_states[1]["state_text"])
            self.assertEqual(transitions[1]["request_state_id"], request_states[1]["request_state_id"])

    def test_masks_hidden_joker_identity(self):
        raw = state("SELECTING_HAND", 0, 1)
        raw["jokers"] = {
            "count": 1,
            "cards": [
                {
                    "key": "j_blueprint",
                    "label": "Blueprint",
                    "state": {"hidden": True},
                }
            ],
        }

        visible = sanitize_visible_state(raw)

        self.assertEqual(
            visible["jokers"]["cards"],
            [{"hidden": True, "state": {"hidden": True}}],
        )

    def test_failed_call_count_gap_keeps_records_but_does_not_guess_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runs"
            run_dir = source / "run-1"
            output = root / "output"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "task.json",
                {"model": {"vendor": "vendor", "name": "model"}, "seed": "AAAAAAA"},
            )
            write_json(
                run_dir / "stats.json",
                {"calls_total": 3, "calls_success": 2, "calls_failed": 1},
            )
            write_json(run_dir / "strategy.json", {})
            with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as output_file:
                for turn in range(1, 4):
                    output_file.write(json.dumps(request(turn)) + "\n")
            with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as output_file:
                for turn in range(1, 4):
                    output_file.write(
                        json.dumps(response(turn, "play", {"cards": [0]})) + "\n"
                    )
            with (run_dir / "gamestates.jsonl").open("w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(state("SELECTING_HAND", 10, 1)) + "\n")
                output_file.write(json.dumps(state("GAME_OVER", 20, 0)) + "\n")

            manifest = import_balatrobench(source_root=source, output_dir=output)

            self.assertEqual(manifest["counts"]["states"], 2)
            self.assertEqual(manifest["counts"]["transitions"], 3)
            self.assertEqual(manifest["counts"]["unaligned_runs"], 1)
            self.assertEqual(manifest["counts"]["bc_candidates"], 0)
            self.assertEqual(manifest["counts"]["text_bc_candidates"], 3)
            transitions = [
                json.loads(line)
                for line in (output / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(item["alignment"] == "unresolved_failed_calls" for item in transitions))
            self.assertTrue(all(item["pre_state_id"] is None for item in transitions))
            self.assertTrue(all(item["post_state_id"] is None for item in transitions))


if __name__ == "__main__":
    unittest.main()
