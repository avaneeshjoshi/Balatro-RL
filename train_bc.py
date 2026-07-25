"""Train a PPO-compatible policy from recorded expert actions."""

import argparse
from pathlib import Path

from ai_agent import train_behavioral_cloning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a behavioral-cloning policy from expert_data.jsonl",
    )
    parser.add_argument(
        "--data",
        default="expert_data.jsonl",
        help="Expert JSONL dataset (default: expert_data.jsonl)",
    )
    parser.add_argument(
        "--model-out",
        default="models/balatro_bc.zip",
        help="Saved SB3 model path (default: models/balatro_bc.zip)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, etc.",
    )
    args = parser.parse_args()

    summary = train_behavioral_cloning(
        data_path=Path(args.data),
        model_path=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    print(
        "Saved best policy to "
        f"{summary['model_path']} "
        f"(epoch {summary['best_epoch']}, "
        f"val_nll={summary['best_validation_loss']:.4f})"
    )


if __name__ == "__main__":
    main()
