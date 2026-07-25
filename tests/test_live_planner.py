import unittest
from pathlib import Path

import torch

from ai_agent.live_planner import (
    bridge_state_to_planner_state,
    load_planner_model,
    predict_live_action,
)
from ai_agent.planner_policy import (
    ACTION_TO_INDEX,
    ACTION_VOCABULARY,
    PlannerPolicy,
    encode_planner_state,
)


def live_state() -> dict:
    return {
        "seq": 10,
        "phase": 4,
        "money": 4,
        "chips": 25,
        "blind_chips": 300,
        "hands_left": 3,
        "discards_left": 2,
        "hand": [
            {
                "index": 1,
                "value": "Ace",
                "suit": "Spades",
                "center": "c_base",
                "seal": "Gold",
                "edition": "foil",
                "debuff": False,
                "facing": "front",
                "extra_chips": 5,
            },
            {
                "index": 2,
                "value": "10",
                "suit": "Hearts",
                "center": "m_bonus",
                "seal": "",
                "edition": "",
                "debuff": True,
                "facing": "front",
            },
        ],
        "hand_levels": {
            "Pair": {"level": 2, "chips": 15, "mult": 3, "played": 4}
        },
        "run": {
            "seed": "VISIBLE-BUT-NOT-ENCODED",
            "ante": 1,
            "round": 2,
            "stake": 1,
            "hands_played": 1,
            "discards_used": 1,
        },
        "blind": {
            "key": "bl_head",
            "name": "The Head",
            "type": "Boss",
            "boss": True,
        },
        "jokers": [],
        "consumables": [],
    }


class LivePlannerTests(unittest.TestCase):
    def test_bridge_state_maps_to_training_contract(self):
        state = bridge_state_to_planner_state(live_state())

        self.assertEqual(state["blind"]["kind"], "BOSS")
        self.assertEqual(state["blind"]["effect"], "All Heart cards are debuffed")
        self.assertEqual(state["stake"], "WHITE")
        self.assertEqual(state["resources"]["hands"]["max"], 4)
        self.assertEqual(state["hand"]["cards"][0]["rank"], "A")
        self.assertEqual(state["hand"]["cards"][0]["suit"], "S")
        self.assertEqual(state["hand"]["cards"][0]["editions"], ["FOIL"])
        self.assertEqual(state["hand"]["cards"][1]["rank"], "T")
        self.assertEqual(state["hand"]["cards"][1]["enhancements"], ["BONUS"])
        self.assertEqual(len(encode_planner_state(state)), 1308)

    def test_saved_baseline_loads_and_returns_legal_action(self):
        model_path = Path(__file__).resolve().parents[1] / "models" / "planner_llm_baseline.pt"
        if not model_path.exists():
            self.skipTest("trained local baseline is not present")
        model, device, checkpoint = load_planner_model(model_path, "cpu")

        prediction = predict_live_action(
            model, bridge_state_to_planner_state(live_state()), device
        )

        self.assertEqual(checkpoint["action_count"], len(ACTION_VOCABULARY))
        self.assertIn(prediction["kind"], {"play", "discard"})
        self.assertTrue(prediction["card_indices"])
        self.assertTrue(set(prediction["card_indices"]) <= {0, 1})

    def test_prediction_masks_illegal_discards(self):
        state = bridge_state_to_planner_state(live_state())
        state["resources"]["discards"]["remaining"] = 0
        model = PlannerPolicy(len(encode_planner_state(state)))

        prediction = predict_live_action(model, state, torch.device("cpu"))

        self.assertEqual(prediction["kind"], "play")

    def test_score_reranker_replaces_lower_pair_with_higher_pair(self):
        raw = live_state()
        raw["blind"] = {
            "key": "bl_small",
            "name": "Small Blind",
            "type": "Small",
            "boss": False,
        }
        raw["hand"] = [
            {"value": rank, "suit": suit, "center": "c_base", "facing": "front"}
            for rank, suit in (("Ace", "Spades"), ("Ace", "Hearts"), ("2", "Clubs"), ("2", "Diamonds"))
        ]
        raw["hand_levels"]["High Card"] = {"level": 1, "chips": 5, "mult": 1}
        state = bridge_state_to_planner_state(raw)
        model = PlannerPolicy(len(encode_planner_state(state)))
        for parameter in model.parameters():
            parameter.data.zero_()
        low_pair = ACTION_TO_INDEX[("play", (2, 3))]
        model.network[-1].bias.data[low_pair] = 10

        prediction = predict_live_action(model, state, torch.device("cpu"))

        self.assertEqual(prediction["policy_action"]["card_indices"], [2, 3])
        self.assertEqual(prediction["card_indices"], [0, 1])
        self.assertEqual(prediction["selection_source"], "score_rerank")
        self.assertEqual(prediction["engine_score"]["hand_type"], "Pair")


if __name__ == "__main__":
    unittest.main()
