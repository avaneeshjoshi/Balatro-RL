"""Run the fixed-vocabulary planner policy through the live Balatro bridge."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_agent.live_planner import (
    bridge_state_to_planner_state,
    load_planner_model,
    predict_live_action,
)
from ai_agent.planner_policy import MAX_HAND_SLOTS
from env import BalatroEnv


DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"


def _summary(raw: dict | None) -> str:
    raw = raw or {}
    return (
        f"chips={raw.get('chips', '?')}/{raw.get('blind_chips', '?')} "
        f"hands={raw.get('hands_left', '?')} discards={raw.get('discards_left', '?')}"
    )


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _reset_after_transition(env: BalatroEnv, retries: int) -> dict | None:
    for attempt in range(retries + 1):
        _, info = env.reset()
        raw = info.get("raw_state")
        if raw is not None:
            return raw
        if attempt < retries:
            print(
                f"Transition still settling; retrying "
                f"({attempt + 1}/{retries})..."
            )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Balatro with the planner baseline")
    parser.add_argument("--model", default="models/planner_llm_baseline.pt")
    parser.add_argument(
        "--bridge-dir",
        default=os.environ.get("BALATRO_BRIDGE_DIR", str(DEFAULT_BRIDGE_DIR)),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--advance-timeout", type=float, default=120.0)
    parser.add_argument("--transition-retries", type=int, default=2)
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Disable exact-score reranking and execute the raw policy action",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-log", default="planner_play_history.jsonl")
    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1 or args.transition_retries < 0:
        raise ValueError(
            "--episodes and --max-steps must be positive; "
            "--transition-retries cannot be negative"
        )
    if args.episodes > 1 and not args.auto_advance:
        raise ValueError("--episodes greater than 1 requires --auto-advance")

    model, device, checkpoint = load_planner_model(args.model, args.device)
    env = BalatroEnv(
        bridge_dir=Path(args.bridge_dir).resolve(),
        state_timeout=15.0,
        advance_timeout=args.advance_timeout,
        auto_advance=args.auto_advance,
        observation_version=2,
        num_card_slots=MAX_HAND_SLOTS,
    )
    log_path = Path(args.run_log).resolve() if args.run_log else None
    print(
        f"Loaded planner epoch {checkpoint.get('best_epoch', '?')} on {device} "
        f"({checkpoint['action_count']} actions)"
    )

    try:
        for episode_number in range(1, args.episodes + 1):
            retries = args.transition_retries if args.auto_advance and episode_number > 1 else 0
            raw = _reset_after_transition(env, retries)
            if raw is None:
                raise RuntimeError(
                    "No playable hand found. Start Balatro or use --auto-advance from a loss/shop screen."
                )
            run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
            blind = raw.get("blind") if isinstance(raw.get("blind"), dict) else {}
            seed = str(run.get("seed") or "unknown")
            print(
                f"Episode {episode_number}/{args.episodes}: {_summary(raw)} "
                f"seed={seed} blind={blind.get('name') or blind.get('key') or '?'}"
            )
            moves = []
            result = "max_steps"
            for step_number in range(1, args.max_steps + 1):
                planner_state = bridge_state_to_planner_state(raw)
                prediction = predict_live_action(
                    model,
                    planner_state,
                    device,
                    stochastic=args.stochastic,
                    temperature=args.temperature,
                    score_plays=not args.policy_only,
                )
                action = np.zeros(1 + MAX_HAND_SLOTS, dtype=np.int64)
                action[0] = 1 if prediction["kind"] == "discard" else 0
                for card_index in prediction["card_indices"]:
                    action[card_index + 1] = 1
                _, reward, terminated, truncated, info = env.step(action)
                sent_cards = info.get("card_indices", [])
                engine_score = prediction.get("engine_score") or {}
                source = " score-reranked" if prediction["selection_source"] == "score_rerank" else ""
                score_text = (
                    f" expected={engine_score['expected_score']:.0f}"
                    if engine_score
                    else ""
                )
                print(
                    f"Step {step_number}: {info.get('action_type', prediction['kind'])} "
                    f"{sent_cards} p={prediction['probability']:.3f}{source}{score_text} "
                    f"reward={reward:.4f} | {_summary(info.get('raw_state'))}"
                )
                post_state = info.get("raw_state") or {}
                moves.append(
                    {
                        "step": step_number,
                        "action": info.get("action_type", prediction["kind"]),
                        "cards": sent_cards,
                        "policy_probability": prediction["probability"],
                        "policy_action": prediction["policy_action"],
                        "selection_source": prediction["selection_source"],
                        "engine_score": prediction["engine_score"],
                        "top_actions": prediction["top_actions"],
                        "hand": planner_state["hand"]["cards"],
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
                raw = post_state
            if log_path is not None:
                _append_log(
                    log_path,
                    {
                        "logged_at": datetime.now(timezone.utc).isoformat(),
                        "model": str(Path(args.model).resolve()),
                        "best_epoch": checkpoint.get("best_epoch"),
                        "deterministic": not args.stochastic,
                        "temperature": args.temperature,
                        "score_plays": not args.policy_only,
                        "seed": seed,
                        "ante": run.get("ante"),
                        "round": run.get("round"),
                        "blind_key": blind.get("key"),
                        "blind_name": blind.get("name"),
                        "blind_target": raw.get("blind_chips"),
                        "result": result,
                        "moves": moves,
                    },
                )
    finally:
        env.close()
    if log_path is not None:
        print(f"Run history: {log_path}")


if __name__ == "__main__":
    main()
