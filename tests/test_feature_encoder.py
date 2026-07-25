import copy
import unittest

import numpy as np

from env.feature_encoder import (
    OBS_DIM_V1,
    OBS_DIM_V2,
    encode_state,
    observation_version_for_dimension,
)


def visible_state() -> dict:
    return {
        "phase": 4,
        "money": 7,
        "chips": 120,
        "blind_chips": 300,
        "hands_left": 3,
        "discards_left": 2,
        "run": {
            "seed": "VISIBLE-BUT-NOT-A-FEATURE",
            "ante": 1,
            "round": 1,
            "stake": 1,
            "deck_remaining": 44,
            "deck_total": 52,
            "hands_played": 1,
            "discards_used": 1,
            "most_played_hand": "Pair",
        },
        "blind": {
            "key": "bl_club",
            "name": "The Club",
            "type": "Boss",
            "boss": True,
            "disabled": False,
            "debuff": {"suit": "Clubs"},
        },
        "hand": [
            {
                "value": "Ace",
                "suit": "Clubs",
                "center": "c_base",
                "seal": "Gold",
                "edition": "foil",
                "debuff": True,
                "facing": "front",
                "forced": False,
                "played_this_ante": False,
            },
            {"value": "King", "suit": "Hearts", "center": "m_bonus"},
        ],
        "hand_levels": {"Pair": {"level": 2, "chips": 20, "mult": 2}},
        "jokers": [
            {
                "key": "j_joker",
                "name": "Joker",
                "edition": "",
                "debuff": False,
                "sell_cost": 1,
                "ability": {"mult": 4},
            }
        ],
        "consumables": [
            {
                "key": "c_pluto",
                "name": "Pluto",
                "set": "Planet",
                "edition": "",
                "debuff": False,
                "sell_cost": 1,
                "ability": {},
            }
        ],
    }


class FeatureEncoderTests(unittest.TestCase):
    def test_v1_shape_is_preserved(self):
        self.assertEqual(encode_state(visible_state(), 1).shape, (OBS_DIM_V1,))
        self.assertEqual(OBS_DIM_V1, 60)

    def test_v2_shape_range_and_finite_values(self):
        observation = encode_state(visible_state(), 2)

        self.assertEqual(observation.shape, (OBS_DIM_V2,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertGreaterEqual(float(observation.min()), 0.0)
        self.assertLessEqual(float(observation.max()), 1.0)

    def test_seed_is_metadata_not_policy_input(self):
        first = visible_state()
        second = copy.deepcopy(first)
        second["run"]["seed"] = "A-COMPLETELY-DIFFERENT-SEED"

        np.testing.assert_array_equal(
            encode_state(first, 2),
            encode_state(second, 2),
        )

    def test_visible_blind_and_card_information_changes_v2(self):
        first = visible_state()
        second = copy.deepcopy(first)
        second["blind"]["key"] = "bl_small"
        second["blind"]["debuff"] = {}
        second["hand"][0]["debuff"] = False

        self.assertFalse(np.array_equal(encode_state(first, 2), encode_state(second, 2)))

    def test_face_down_card_identity_is_not_exposed(self):
        first = visible_state()
        first["hand"][0]["facing"] = "back"
        second = copy.deepcopy(first)
        second["hand"][0].update(
            {
                "value": "2",
                "suit": "Spades",
                "center": "m_glass",
                "seal": "Purple",
                "edition": "polychrome",
                "debuff": False,
            }
        )

        np.testing.assert_array_equal(
            encode_state(first, 2),
            encode_state(second, 2),
        )

    def test_observation_version_can_be_inferred_from_dimension(self):
        self.assertEqual(observation_version_for_dimension(OBS_DIM_V1), 1)
        self.assertEqual(observation_version_for_dimension(OBS_DIM_V2), 2)


if __name__ == "__main__":
    unittest.main()
