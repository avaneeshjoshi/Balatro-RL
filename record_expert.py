"""
Recorder: collect (observation, action) pairs while you play for Behavioral Cloning.

You play through this script: it shows the current hand, you type an action
(play/discard + card indices), it sends the command to the game and logs the pair to .jsonl.

Usage:
  1. Start Balatro and get to a hand-selection screen.
  2. Run: python record_expert.py [--output expert_data.jsonl]
  3. When state appears, type e.g. "play 1,2,3" or "discard 4,5" (1-based indices).
  4. Type "quit" or Ctrl+C to stop.

Each line is versioned JSON containing `obs`, `action`, and the structured
`raw_state`. New files use observation v2; an existing v1 file stays v1.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from env import BalatroEnv
from env.balatro_env import MAX_PLAY_CARDS, NUM_CARD_SLOTS
from env.feature_encoder import observation_version_for_dimension

DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"


def existing_observation_version(path: Path) -> int | None:
    """Infer an existing JSONL file's observation version from its first record."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                return observation_version_for_dimension(len(record["obs"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot determine observation version from {path}:{line_number}"
                ) from exc
    return None


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
    parser.add_argument(
        "--observation-version",
        type=int,
        choices=(1, 2),
        help="Encoder version. Existing output files are detected; new files default to 2.",
    )
    args = parser.parse_args()
    bridge_path = Path(args.bridge_dir).resolve()
    out_path = Path(args.output)
    existing_version = existing_observation_version(out_path)
    if existing_version is not None and args.observation_version not in (None, existing_version):
        raise ValueError(
            f"{out_path} contains observation v{existing_version}; choose another output file "
            f"for observation v{args.observation_version}"
        )
    observation_version = args.observation_version or existing_version or 2

    env = BalatroEnv(
        bridge_dir=bridge_path,
        state_timeout=15.0,
        observation_version=observation_version,
    )
    print(f"Bridge: {bridge_path}")
    print(f"Output: {out_path.absolute()}")
    print(f"Observation: v{observation_version} ({env.obs_dim} values)")
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
            print(
                f"  chips={raw.get('chips')}/{raw.get('blind_chips')} "
                f"hands_left={raw.get('hands_left')} discards_left={raw.get('discards_left')}"
            )
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
            if int(action[0]) == 1 and int(raw.get("discards_left", 0)) <= 0:
                print("Invalid input. No discards remain; choose a play action.")
                continue
            print("Sending command...")
            _, _, terminated, truncated, step_info = env.step(action)
            if truncated:
                reason = step_info.get("truncated_reason") or "unknown"
                print(f"Action was not recorded because the bridge stopped: {reason}")
                continue
            if step_info.get("action_overridden"):
                print("Action was changed by the environment and was not recorded.")
                continue
            obs_list = obs.tolist()
            action_list = action.tolist()
            f.write(
                json.dumps(
                    {
                        "schema_version": 3,
                        "observation_version": observation_version,
                        "obs": obs_list,
                        "action": action_list,
                        "raw_state": raw,
                    }
                )
                + "\n"
            )
            f.flush()
            count += 1
            print(f"Recorded #{count}.")
            if terminated:
                reason = step_info.get("terminated_reason") or "blind_finished"
                print(f"Round ended: {reason}")
            print()

    env.close()
    print(f"Recorded {count} pairs to {out_path.absolute()}")


if __name__ == "__main__":
    main()
