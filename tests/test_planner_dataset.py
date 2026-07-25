import json
import tempfile
import unittest
from pathlib import Path

from ai_agent.planner_dataset import (
    audit_planner_dataset,
    build_planner_dataset,
    parse_request_state,
    validate_against_structured,
)


STATE_TEXT = """# Current Game State

- **Phase**: Playing Phase (gamestate is SELECTING_HAND)
- **Round**: 3
- **Ante**: 1/8
- **Money**: $7
- **Hands left**: 2/4
- **Discards left**: 1/3
- **Current Blind**: Boss (The Hook: Discards 2 random cards per hand played)
- **Target Score**: 600
- **Current Score**: 108
- **Deck**: RED
- **Stake**: WHITE

## Jokers

The current Jokers count is 1/5.

- 0: Zany Joker (+12 Mult if played hand contains a Three of a Kind)
  - **Sell value**: $2

## Consumables

The current Consumables count is 1/2.

- 0: The Fool (Creates the last Tarot or Planet card used during this run)
  - **Sell value**: $1

## Poker Hands

{poker_hands}

## Current Hand

The current card count is 4 / 8

- 0: A of D (`D_A`)
  - **BONUS Enhancement**
  - **FOIL Edition**
  - **PURPLE Seal**
- 1: the card is face down
- 2: this is a stone card (no suit and no rank)
- 3: 7 of H (`H_7`)
  - **Debuff**: this card is debuffed and will not be counted in scoring
"""


def poker_hand_text() -> str:
    names = [
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Flush",
        "Straight",
        "Three of a Kind",
        "Two Pair",
        "Pair",
        "High Card",
    ]
    return "\n".join(
        f"- **{name}** (Level 1):\n"
        "  - **Chips**: 10\n"
        "  - **Mult**: 2\n"
        f"  - During this run you have played {name} 0 times, 0 of which were played this round."
        for name in names
    )


def rendered_state() -> str:
    return STATE_TEXT.format(poker_hands=poker_hand_text())


def structured_state() -> dict:
    hands = {
        name: {"level": 1, "chips": 10, "mult": 2, "played": 0, "played_this_round": 0}
        for name in [
            "Flush Five",
            "Flush House",
            "Five of a Kind",
            "Straight Flush",
            "Four of a Kind",
            "Full House",
            "Flush",
            "Straight",
            "Three of a Kind",
            "Two Pair",
            "Pair",
            "High Card",
        ]
    }
    return {
        "state": "SELECTING_HAND",
        "round_num": 3,
        "ante_num": 1,
        "money": 7,
        "deck": "RED",
        "stake": "WHITE",
        "round": {"chips": 108, "hands_left": 2, "discards_left": 1},
        "hand": {
            "count": 4,
            "cards": [
                {
                    "key": "D_A",
                    "label": "Bonus Card",
                    "modifier": {
                        "enhancement": "BONUS",
                        "edition": "FOIL",
                        "seal": "PURPLE",
                    },
                },
                {"hidden": True},
                {"label": "Stone Card"},
                {"key": "H_7", "label": "Base Card", "state": {"debuff": True}},
            ],
        },
        "jokers": {
            "count": 1,
            "cards": [
                {
                    "label": "Zany Joker",
                    "value": {"effect": "+12 Mult if played hand contains a Three of a Kind"},
                }
            ],
        },
        "consumables": {
            "count": 1,
            "cards": [
                {
                    "label": "The Fool",
                    "value": {"effect": "Creates the last Tarot or Planet card used during this run"},
                }
            ],
        },
        "hands": hands,
        "blinds": {
            "boss": {
                "status": "CURRENT",
                "name": "The Hook",
                "effect": "Discards 2 random cards per hand played",
                "score": 600,
            }
        },
    }


class PlannerDatasetTests(unittest.TestCase):
    def test_parses_visible_state_and_card_modifiers(self):
        state, errors = parse_request_state(rendered_state())

        self.assertEqual(errors, [])
        self.assertEqual(state["blind"]["name"], "The Hook")
        self.assertEqual(state["blind"]["target_score"], 600)
        self.assertEqual(state["jokers"]["cards"][0]["name"], "Zany Joker")
        self.assertFalse(state["jokers"]["cards"][0]["hidden"])
        self.assertEqual(len(state["poker_hands"]), 12)
        self.assertEqual(state["hand"]["cards"][0]["enhancements"], ["BONUS"])
        self.assertEqual(state["hand"]["cards"][0]["editions"], ["FOIL"])
        self.assertEqual(state["hand"]["cards"][0]["seals"], ["PURPLE"])
        self.assertTrue(state["hand"]["cards"][1]["hidden"])
        self.assertTrue(state["hand"]["cards"][2]["stone"])
        self.assertTrue(state["hand"]["cards"][3]["debuffed"])
        self.assertIsNone(state["hand"]["cards"][0]["extra_chips"])

    def test_validates_rendered_state_against_structured_overlap(self):
        state, errors = parse_request_state(rendered_state())
        checks = validate_against_structured(state, structured_state())

        self.assertEqual(errors, [])
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()), checks)

    def test_builds_joined_examples_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "run-1"
            request_id = f"{run_id}#request-state-00001"
            state_id = f"{run_id}#state-00001"
            transition_id = f"{run_id}#transition-00002"
            files = {
                "runs.jsonl": {
                    "run_id": run_id,
                    "task": {
                        "model": {"vendor": "vendor", "name": "model"},
                        "seed": "AAAAAAA",
                        "deck": "RED",
                        "stake": "WHITE",
                    },
                    "stats": {"run_won": True, "final_ante": 2, "final_round": 4},
                    "strategy": {"name": "Default"},
                },
                "request_states.jsonl": {
                    "request_state_id": request_id,
                    "state_text": rendered_state(),
                },
                "states.jsonl": {
                    "state_id": state_id,
                    "visible_state": structured_state(),
                },
                "transitions.jsonl": {
                    "transition_id": transition_id,
                    "run_id": run_id,
                    "turn": 2,
                    "request_state_id": request_id,
                    "pre_state_id": state_id,
                    "text_bc_candidate": True,
                    "action": {"name": "play", "arguments": {"cards": [0, 3]}},
                    "score_delta": 120,
                    "round_result": None,
                    "source": {"response_line": 2},
                },
            }
            for name, record in files.items():
                (root / name).write_text(json.dumps(record) + "\n", encoding="utf-8")

            manifest = build_planner_dataset(input_dir=root)
            example = json.loads((root / "planner_examples.jsonl").read_text(encoding="utf-8"))
            audit = audit_planner_dataset(dataset_path=root / "planner_examples.jsonl")

            self.assertEqual(manifest["counts"]["examples"], 1)
            self.assertEqual(manifest["counts"]["usable_examples"], 1)
            self.assertEqual(manifest["counts"]["exact_full_matches"], 1)
            self.assertEqual(example["provenance"]["seed"], "AAAAAAA")
            self.assertEqual(example["action"]["card_indices"], [0, 3])
            self.assertEqual([card["key"] for card in example["action"]["selected_cards"]], ["D_A", "H_7"])
            self.assertEqual(example["state"]["hand"]["cards"][0]["extra_chips"], 0)
            self.assertEqual(example["quality"]["validation_mismatches"], [])
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["integrity_errors"], 0)


if __name__ == "__main__":
    unittest.main()
