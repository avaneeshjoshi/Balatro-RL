import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from env.balatro_env import ACTION_DIM, BalatroEnv
from run_env_example import sample_valid_action


def make_state(**overrides):
    state = {
        "seq": 1,
        "phase": 4,
        "money": 4,
        "chips": 36,
        "blind_chips": 300,
        "hands_left": 3,
        "discards_left": 2,
        "hand": [{"index": 1, "value": "A", "suit": "Spades"}],
        "hand_levels": {},
        "last_hand_played": "",
        "round_result": "",
    }
    state.update(overrides)
    return state


def make_action(action_type: int = 0) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.int64)
    action[0] = action_type
    action[1] = 1
    return action


class BalatroEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = BalatroEnv(
            bridge_dir=Path(self.temp_dir.name),
            step_delay=0,
        )

    def tearDown(self):
        self.env.close()
        self.temp_dir.cleanup()

    def test_random_sampler_only_plays_when_no_discards_remain(self):
        for _ in range(100):
            action = sample_valid_action(hand_size=8, discards_left=0)
            self.assertEqual(action[0], 0)
            self.assertGreaterEqual(action[1:].sum(), 1)
            self.assertLessEqual(action[1:].sum(), 5)

    def test_env_converts_impossible_discard_to_play(self):
        self.env._last_raw = make_state(discards_left=0)

        action_type, cards = self.env._action_to_command(make_action(1))

        self.assertEqual(action_type, "play")
        self.assertEqual(cards, [1])

    def test_env_preserves_legal_discard(self):
        self.env._last_raw = make_state(discards_left=1)

        action_type, cards = self.env._action_to_command(make_action(1))

        self.assertEqual(action_type, "discard")
        self.assertEqual(cards, [1])

    def test_env_can_send_planner_slots_beyond_eight(self):
        planner_env = BalatroEnv(
            bridge_dir=Path(self.temp_dir.name),
            step_delay=0,
            num_card_slots=12,
        )
        planner_env._last_raw = make_state(
            hand=[{"index": index} for index in range(1, 11)]
        )
        action = np.zeros(13, dtype=np.int64)
        action[9] = 1
        action[10] = 1

        action_type, cards = planner_env._action_to_command(action)

        planner_env.close()
        self.assertEqual(action_type, "play")
        self.assertEqual(cards, [9, 10])

    def test_stale_hand_name_does_not_repeat_bonus(self):
        self.env._last_raw = make_state(last_hand_played="Pair")
        self.env._last_chips = 36
        unchanged = make_state(seq=2, last_hand_played="Pair")

        play_reward = self.env._compute_reward(unchanged, "play")
        discard_reward = self.env._compute_reward(unchanged, "discard")

        self.assertEqual(play_reward, 0.0)
        self.assertEqual(discard_reward, -self.env.reward_discard_penalty)

    def test_scored_play_gets_hand_bonus_once(self):
        self.env._last_raw = make_state()
        self.env._last_chips = 36
        scored = make_state(
            seq=2,
            chips=46,
            hands_left=2,
            last_hand_played="Pair",
        )

        reward = self.env._compute_reward(scored, "play")

        self.assertAlmostEqual(reward, 0.006)

    def test_step_terminates_on_won_blind(self):
        self.env._last_raw = make_state(seq=1)
        self.env._last_chips = 36
        self.env._last_seq = 1
        terminal = make_state(
            seq=2,
            chips=336,
            hands_left=2,
            hand=[],
            last_hand_played="High Card",
            round_result="won",
        )

        with (
            patch.object(self.env, "_write_command") as write_command,
            patch.object(
                self.env,
                "_wait_for_new_state",
                return_value=(terminal, "new_state"),
            ),
        ):
            _, reward, terminated, truncated, info = self.env.step(make_action())

        write_command.assert_called_once_with("play", [1])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminated_reason"], "blind_won")
        self.assertAlmostEqual(reward, 1.03)

    def test_step_terminates_on_lost_blind(self):
        self.env._last_raw = make_state(seq=1, hands_left=1)
        self.env._last_chips = 36
        self.env._last_seq = 1
        terminal = make_state(
            seq=2,
            hands_left=0,
            hand=[],
            last_hand_played="High Card",
            round_result="lost",
        )

        with (
            patch.object(self.env, "_write_command"),
            patch.object(
                self.env,
                "_wait_for_new_state",
                return_value=(terminal, "new_state"),
            ),
        ):
            _, reward, terminated, truncated, info = self.env.step(make_action())

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["terminated_reason"], "blind_lost")
        self.assertAlmostEqual(reward, -1.0)

    def test_reset_requests_advance_after_terminal_state(self):
        self.env.close()
        self.env = BalatroEnv(
            bridge_dir=Path(self.temp_dir.name),
            step_delay=0,
            auto_advance=True,
        )
        terminal = make_state(
            seq=2,
            hands_left=0,
            hand=[],
            round_result="lost",
        )
        playable = make_state(seq=3, chips=0, hands_left=4)

        with (
            patch.object(self.env, "_wait_for_state", return_value=terminal),
            patch.object(self.env, "_read_state", return_value=playable),
            patch.object(self.env, "_write_command") as write_command,
        ):
            observation, info = self.env.reset()

        write_command.assert_called_once_with("advance", [])
        self.assertEqual(observation.shape, (60,))
        self.assertEqual(info["raw_state"], playable)


if __name__ == "__main__":
    unittest.main()
