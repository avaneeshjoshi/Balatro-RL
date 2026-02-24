"""
Recorder: collect (observation, action) pairs while you play for Behavioral Cloning.

You play through this script: it shows the current hand, you type an action
(play/discard + card indices), it sends the command to the game and logs the pair to .jsonl.

Usage:
  1. Start Balatro and get to a hand-selection screen.
  2. Run: python record_expert.py [--output expert_data.jsonl]
  3. When state appears, type e.g. "play 1,2,3" or "discard 4,5" (1-based indices).
  4. Type "quit" or Ctrl+C to stop.

Each line in the output file is JSON: {"obs": [...], "action": [...]} with the same
format the env uses (obs = 30-dim float vector, action = 9-dim int: play/discard + 8 card bits).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from env import BalatroEnv
from env.balatro_env import MAX_PLAY_CARDS, NUM_CARD_SLOTS

DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"


def parse_action_string(s: str, hand_size: int) -> np.ndarray | None:
    """
    Parse "play 1,2,3" or "discard 4,5" into MultiDiscrete action.
    Returns None if invalid. Indices are 1-based; hand_size caps valid indices.
    """
    s = s.strip().lower()
    if not s:
        return None
    parts = s.split(maxsplit=1)
    if len(parts) < 2:
        return None
    kind, rest = parts[0], parts[1]
    if kind in ("p", "play"):
        play = 0
        max_cards = min(MAX_PLAY_CARDS, hand_size)
    elif kind in ("d", "discard"):
        play = 1
        max_cards = hand_size
    else:
        return None
    try:
        indices = [int(x.strip()) for x in rest.replace(",", " ").split() if x.strip()]
    except ValueError:
        return None
    if not indices:
        return None
    # Clamp to valid 1-based indices and dedupe
    indices = sorted(set(i for i in indices if 1 <= i <= hand_size))[:max_cards]
    if not indices:
        return None
    action = np.zeros(1 + NUM_CARD_SLOTS, dtype=np.int64)
    action[0] = play
    for i in indices:
        action[i] = 1  # 1-based index into action[1..8]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Record expert (obs, action) pairs for Behavioral Cloning.")
    parser.add_argument(
        "--output", "-o",
        default="expert_data.jsonl",
        help="Output .jsonl file (default: expert_data.jsonl)",
    )
    parser.add_argument(
        "--bridge-dir",
        default=os.environ.get("BALATRO_BRIDGE_DIR", str(DEFAULT_BRIDGE_DIR)),
        help="Bridge directory (state.json / command.json)",
    )
    args = parser.parse_args()
    bridge_path = Path(args.bridge_dir).resolve()
    out_path = Path(args.output)

    env = BalatroEnv(bridge_dir=bridge_path, state_timeout=15.0)
    print(f"Bridge: {bridge_path}")
    print(f"Output: {out_path.absolute()}")
    print("When you see a hand, type:  play 1,2,3   or   discard 4,5   (1-based indices). Type 'quit' to stop.\n")

    count = 0
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            obs, info = env.reset()
            raw = info.get("raw_state")
            if raw is None:
                print("(waiting for state...)")
                continue
            hand = raw.get("hand") or []
            hand_size = len(hand)
            if hand_size == 0:
                print("(hand empty, waiting for next state...)")
                continue
            print(f"Hand ({hand_size} cards):")
            for i, c in enumerate(hand, 1):
                print(f"  {i}: {c.get('value', '?')} of {c.get('suit', '?')}")
            print(f"  chips={raw.get('chips')} hands_left={raw.get('hands_left')} discards_left={raw.get('discards_left')}")
            try:
                line = input("Action (play/discard + indices)> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                break
            if line.lower() in ("quit", "q", "exit"):
                print("Stopped.")
                break
            action = parse_action_string(line, hand_size)
            if action is None:
                print("Invalid input. Use e.g. 'play 1,2,3' or 'discard 4,5' (1-based, max 5 for play).")
                continue
            obs_list = obs.tolist()
            action_list = action.tolist()
            f.write(json.dumps({"obs": obs_list, "action": action_list}) + "\n")
            f.flush()
            count += 1
            print(f"Recorded #{count}. Sending command...")
            env.step(action)
            print()

    env.close()
    print(f"Recorded {count} pairs to {out_path.absolute()}")


if __name__ == "__main__":
    main()
