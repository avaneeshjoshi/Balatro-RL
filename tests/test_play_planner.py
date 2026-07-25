import unittest

from play_planner import _reset_after_transition


class FakeEnv:
    def __init__(self, states):
        self.states = iter(states)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return None, {"raw_state": next(self.states)}


class PlayPlannerTests(unittest.TestCase):
    def test_transition_reset_retries_until_playable(self):
        playable = {"hand": [{"index": 1}]}
        env = FakeEnv([None, playable])

        result = _reset_after_transition(env, retries=2)

        self.assertIs(result, playable)
        self.assertEqual(env.reset_calls, 2)

    def test_transition_reset_stops_at_retry_limit(self):
        env = FakeEnv([None, None])

        result = _reset_after_transition(env, retries=1)

        self.assertIsNone(result)
        self.assertEqual(env.reset_calls, 2)


if __name__ == "__main__":
    unittest.main()
