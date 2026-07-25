import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ai_agent.planner_policy import (
    ACTION_TO_INDEX,
    ACTION_VOCABULARY,
    PlannerPolicy,
    PreparedPlannerDataset,
    build_action_vocabulary,
    encode_planner_state,
    legal_action_mask,
)


def sample_state() -> dict:
    cards = []
    for index, (rank, suit) in enumerate((("A", "S"), ("K", "H"), ("2", "D"))):
        cards.append(
            {
                "index": index,
                "hidden": False,
                "stone": False,
                "rank": rank,
                "suit": suit,
                "enhancements": [],
                "editions": [],
                "seals": [],
                "debuffed": False,
                "extra_chips": 0,
            }
        )
    return {
        "phase": "SELECTING_HAND",
        "round": 1,
        "ante": {"current": 1, "max": 8},
        "money": 4,
        "resources": {
            "hands": {"remaining": 4, "max": 4},
            "discards": {"remaining": 0, "max": 3},
        },
        "blind": {
            "kind": "SMALL",
            "name": "Small Blind",
            "effect": "",
            "target_score": 300,
            "current_score": 0,
        },
        "deck": "RED",
        "stake": "WHITE",
        "poker_hands": {},
        "hand": {"count": 3, "limit": 8, "cards": cards},
        "jokers": {"count": 0, "limit": 5, "cards": []},
        "consumables": {"count": 0, "limit": 2, "cards": []},
    }


class PlannerPolicyTests(unittest.TestCase):
    def test_action_vocabulary_contains_every_one_to_five_card_set(self):
        self.assertEqual(len(build_action_vocabulary(8)), 436)
        self.assertEqual(len(ACTION_VOCABULARY), 3170)

    def test_encoder_is_finite_and_seed_independent(self):
        state = sample_state()
        encoded = encode_planner_state(state)
        state["seed"] = "SHOULD_NOT_BE_USED"
        encoded_with_seed = encode_planner_state(state)
        self.assertTrue(np.isfinite(encoded).all())
        np.testing.assert_array_equal(encoded, encoded_with_seed)

    def test_legal_mask_excludes_discards_and_cards_outside_hand(self):
        mask = legal_action_mask(sample_state())
        self.assertTrue(mask[ACTION_TO_INDEX[("play", (0, 1, 2))]])
        self.assertFalse(mask[ACTION_TO_INDEX[("play", (0, 3))]])
        self.assertFalse(mask[ACTION_TO_INDEX[("discard", (0,))]])

    def test_dataset_and_policy_forward(self):
        state = sample_state()
        record = {
            "state": state,
            "action_targets": [
                {
                    "kind": "play",
                    "card_indices": [0],
                    "target_probability": 1.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = PreparedPlannerDataset(path)
            model = PlannerPolicy(dataset.observation_dimension)
            logits = model(dataset.observations)
            masked = logits.masked_fill(~dataset.legal_masks, -1e9)
            loss = -(dataset.targets * torch.log_softmax(masked, dim=1)).sum()
            loss.backward()
            self.assertEqual(logits.shape, (1, len(ACTION_VOCABULARY)))
            self.assertTrue(torch.isfinite(loss))

    def test_trained_checkpoint_records_comparison_metadata(self):
        from ai_agent.planner_policy import train_planner_policy

        record = {
            "state": sample_state(),
            "action_targets": [
                {"kind": "play", "card_indices": [0], "target_probability": 1.0}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            train_path = directory_path / "train.jsonl"
            validation_path = directory_path / "validation.jsonl"
            model_path = directory_path / "model.pt"
            line = json.dumps(record) + "\n"
            train_path.write_text(line, encoding="utf-8")
            validation_path.write_text(line, encoding="utf-8")
            train_planner_policy(
                train_path=train_path,
                validation_path=validation_path,
                model_path=model_path,
                epochs=1,
                batch_size=1,
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertEqual(checkpoint["action_count"], len(ACTION_VOCABULARY))
        self.assertEqual(checkpoint["training_config"]["epochs"], 1)
        self.assertIn("nll", checkpoint["validation_metrics"])


if __name__ == "__main__":
    unittest.main()
