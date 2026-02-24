"""
Example: run the Balatro Gymnasium env with a random policy for a few steps.
Set BALATRO_BRIDGE_DIR to the bridge folder path if not using the default below.
Uses valid-action sampling (hand size and Balatro 5-card play limit) instead of raw action_space.sample().
"""
import os
import random
import time
from pathlib import Path

import numpy as np

from env import BalatroEnv
from env.balatro_env import MAX_PLAY_CARDS, NUM_CARD_SLOTS

# Default: repo bridge folder (same path the Lua bridge writes state.json to)
DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"
BRIDGE_DIR = os.environ.get("BALATRO_BRIDGE_DIR", str(DEFAULT_BRIDGE_DIR))


def _state_summary(raw: dict | None) -> str:
    if raw is None:
        return "no state"
    hand = raw.get("hand") or []
    return (
        f"chips={raw.get('chips', '?')} hands_left={raw.get('hands_left', '?')} "
        f"discards_left={raw.get('discards_left', '?')} hand_size={len(hand)}"
    )


def _hand_str(raw: dict | None) -> str:
    if raw is None:
        return "no state"
    hand = raw.get("hand") or []
    if not hand:
        return "hand: (empty)"
    parts = [f"{c.get('value', '?')} of {c.get('suit', '?')}" for c in hand]
    return "hand: " + ", ".join(parts)


def sample_valid_action(hand_size: int) -> np.ndarray:
    """
    Sample a random action that respects hand size and Balatro rules:
    - Play: 1 to min(5, hand_size) cards; indices only from 1..hand_size.
    - Discard: 1 to hand_size cards; indices only from 1..hand_size.
    """
    hand_size = max(1, min(hand_size, NUM_CARD_SLOTS))
    play_or_discard = random.randint(0, 1)  # 0 = play, 1 = discard
    if play_or_discard == 0:
        # Play: at most MAX_PLAY_CARDS (5), at least 1
        n_cards = random.randint(1, min(MAX_PLAY_CARDS, hand_size))
    else:
        # Discard: 1 to hand_size
        n_cards = random.randint(1, hand_size)
    indices_1based = set(random.sample(range(1, hand_size + 1), n_cards))
    action = np.zeros(1 + NUM_CARD_SLOTS, dtype=np.int64)
    action[0] = play_or_discard
    for i in range(NUM_CARD_SLOTS):
        if (i + 1) in indices_1based:
            action[i + 1] = 1
    return action


def main() -> None:
    bridge_path = Path(BRIDGE_DIR).resolve()
    state_file = bridge_path / "state.json"
    print(f"Watching bridge at: {bridge_path}")
    print(f"Initial state.json exists: {state_file.exists()}")

    env = BalatroEnv(bridge_dir=bridge_path, state_timeout=15.0)
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "| Sample:", obs[:8])
    print("Reset state:", _state_summary(info.get("raw_state")))
    print("  ", _hand_str(info.get("raw_state")))
    for t in range(20):
        raw = info.get("raw_state")
        hand_size = len((raw or {}).get("hand") or [])
        if hand_size == 0:
            hand_size = NUM_CARD_SLOTS
        action = sample_valid_action(hand_size)
        obs, reward, term, trunc, info = env.step(action)
        summary = _state_summary(info.get("raw_state"))
        print(f"Step {t+1}: reward={reward:.6f} term={term} trunc={trunc} | {summary}")
        print("  ", _hand_str(info.get("raw_state")))
        if term or trunc:
            why = info.get("terminated_reason") or info.get("truncated_reason") or "unknown"
            print(f"Stopped: terminated={term} truncated={trunc} reason={why}")
            break
        time.sleep(1)
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
