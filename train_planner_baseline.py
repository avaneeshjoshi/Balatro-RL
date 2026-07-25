"""Train the pre-rollout LLM-imitation planner baseline."""

import argparse
import json
from pathlib import Path

from ai_agent.planner_policy import train_planner_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the planner action-ranking baseline")
    parser.add_argument(
        "--train",
        default="data/balatrobench/training/planner_train.jsonl",
    )
    parser.add_argument(
        "--validation",
        default="data/balatrobench/training/planner_validation.jsonl",
    )
    parser.add_argument("--model-out", default="models/planner_llm_baseline.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_planner_policy(
        train_path=Path(args.train),
        validation_path=Path(args.validation),
        model_path=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
