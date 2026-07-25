"""Run a saved behavioral-cloning policy through the Balatro bridge."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO

from env import BalatroEnv
from env.feature_encoder import observation_version_for_dimension

DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"


def state_summary(raw: dict | None) -> str:
    if raw is None:
        return "no playable state"
    return (
        f"chips={raw.get('chips', '?')}/{raw.get('blind_chips', '?')} "
        f"hands={raw.get('hands_left', '?')} "
        f"discards={raw.get('discards_left', '?')}"
    )


def run_context(raw: dict | None) -> dict:
    raw = raw or {}
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    blind = raw.get("blind") if isinstance(raw.get("blind"), dict) else {}
    return {
        "seed": str(run.get("seed") or "unknown"),
        "ante": run.get("ante"),
        "round": run.get("round"),
        "stake": run.get("stake"),
        "blind_key": blind.get("key"),
        "blind_name": blind.get("name"),
    }


def append_run_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a Balatro blind with a cloned policy",
    )
    parser.add_argument(
        "--model",
        default="models/balatro_bc.zip",
        help="Saved SB3 model (default: models/balatro_bc.zip)",
    )
    parser.add_argument(
        "--bridge-dir",
        default=os.environ.get(
            "BALATRO_BRIDGE_DIR",
            str(DEFAULT_BRIDGE_DIR),
        ),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Skip shops, select the next blind, and restart after losses",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of selecting the most likely action",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--run-log",
        default="play_history.jsonl",
        help="Append seeds, outcomes, and moves to this JSONL file (empty disables logging)",
    )
    args = parser.parse_args()

    bridge_path = Path(args.bridge_dir).resolve()
    model = PPO.load(args.model, device=args.device)
    model_observation_dim = int(model.observation_space.shape[0])
    observation_version = observation_version_for_dimension(model_observation_dim)
    env = BalatroEnv(
        bridge_dir=bridge_path,
        state_timeout=15.0,
        auto_advance=args.auto_advance,
        observation_version=observation_version,
    )
    model.set_env(env)
    run_log_path = Path(args.run_log).resolve() if args.run_log else None
    print(
        f"Loaded observation v{observation_version} "
        f"({model_observation_dim} values)"
    )

    if args.episodes < 1:
        env.close()
        raise ValueError("--episodes must be at least 1")
    if args.episodes > 1 and not args.auto_advance:
        env.close()
        raise ValueError("--episodes greater than 1 requires --auto-advance")

    for episode_number in range(1, args.episodes + 1):
        observation, info = env.reset()
        if info.get("raw_state") is None:
            env.close()
            raise RuntimeError(
                "No playable hand found. Open a blind in Balatro and run again."
            )
        print(
            f"Episode {episode_number}/{args.episodes}: "
            f"{state_summary(info['raw_state'])}"
        )
        context = run_context(info["raw_state"])
        initial_blind_target = info["raw_state"].get("blind_chips")
        print(
            f"  seed={context['seed']} ante={context['ante']} "
            f"blind={context['blind_name'] or context['blind_key'] or '?'}"
        )
        moves: list[dict] = []
        result = "max_steps"

        for step_number in range(1, args.max_steps + 1):
            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            action_type = info.get("action_type", "?")
            card_indices = info.get("card_indices", [])
            override = " (discard converted to play)" if info.get("action_overridden") else ""
            print(
                f"Step {step_number}: {action_type} {card_indices}{override} "
                f"reward={reward:.4f} | {state_summary(info.get('raw_state'))}"
            )
            post_state = info.get("raw_state") or {}
            moves.append(
                {
                    "step": step_number,
                    "action": action_type,
                    "cards": card_indices,
                    "reward": float(reward),
                    "chips": post_state.get("chips"),
                    "hands_left": post_state.get("hands_left"),
                    "discards_left": post_state.get("discards_left"),
                }
            )
            if terminated or truncated:
                result = (
                    info.get("terminated_reason")
                    or info.get("truncated_reason")
                    or "unknown"
                )
                print(f"Stopped: {result}")
                break

        if run_log_path is not None:
            final_state = info.get("raw_state") or {}
            append_run_log(
                run_log_path,
                {
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "model": str(Path(args.model).resolve()),
                    "observation_version": observation_version,
                    "deterministic": not args.stochastic,
                    **context,
                    "blind_target": final_state.get("blind_chips", initial_blind_target),
                    "result": result,
                    "moves": moves,
                },
            )

    env.close()
    if run_log_path is not None:
        print(f"Run history: {run_log_path}")


if __name__ == "__main__":
    main()
