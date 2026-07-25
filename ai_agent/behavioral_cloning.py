"""Behavioral cloning for the Balatro Stable Baselines3 policy."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from env.balatro_env import ACTION_DIM
from env.feature_encoder import (
    OBSERVATION_DIMS,
    observation_version_for_dimension,
)


class ExpertDataset(Dataset):
    """In-memory expert observation/action pairs."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])


class OfflineBalatroEnv(gym.Env):
    """Space-only environment used to construct an SB3 policy offline."""

    metadata = {"render_modes": []}

    def __init__(self, observation_dim: int):
        self.observation_dim = observation_dim
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([2] * ACTION_DIM)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(self.observation_dim, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        raise RuntimeError("OfflineBalatroEnv does not execute game actions")


@dataclass
class BCMetrics:
    loss: float
    field_accuracy: float
    action_type_accuracy: float
    card_accuracy: float
    exact_action_accuracy: float


def load_expert_data(path: str | Path) -> ExpertDataset:
    """Load and validate JSONL records written by record_expert.py."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Expert dataset not found: {data_path}")

    observations: list[list[float]] = []
    actions: list[list[int]] = []
    observation_dim: int | None = None
    with data_path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{data_path}:{line_number}: invalid JSON"
                ) from exc

            observation = record.get("obs")
            action = record.get("action")
            if not isinstance(observation, list):
                raise ValueError(
                    f"{data_path}:{line_number}: obs must contain a numeric list"
                )
            if observation_dim is None:
                if len(observation) not in OBSERVATION_DIMS.values():
                    expected = sorted(OBSERVATION_DIMS.values())
                    raise ValueError(
                        f"{data_path}:{line_number}: obs must contain one of "
                        f"the supported dimensions {expected}"
                    )
                observation_dim = len(observation)
            elif len(observation) != observation_dim:
                raise ValueError(
                    f"{data_path}:{line_number}: obs must contain {observation_dim} values; "
                    "the dataset mixes observation versions"
                )
            if not isinstance(action, list) or len(action) != ACTION_DIM:
                raise ValueError(
                    f"{data_path}:{line_number}: action must contain {ACTION_DIM} values"
                )
            try:
                observation_values = [float(value) for value in observation]
                action_values = [int(value) for value in action]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{data_path}:{line_number}: obs/action values must be numeric"
                ) from exc
            if not np.isfinite(observation_values).all():
                raise ValueError(
                    f"{data_path}:{line_number}: obs contains a non-finite value"
                )
            if any(value not in (0, 1) for value in action_values):
                raise ValueError(
                    f"{data_path}:{line_number}: action values must be 0 or 1"
                )
            if sum(action_values[1:]) == 0:
                raise ValueError(
                    f"{data_path}:{line_number}: action must select at least one card"
                )
            if action_values[0] == 0 and sum(action_values[1:]) > 5:
                raise ValueError(
                    f"{data_path}:{line_number}: play action selects more than 5 cards"
                )
            observations.append(observation_values)
            actions.append(action_values)

    if len(observations) < 2:
        raise ValueError("Expert dataset must contain at least 2 records")
    return ExpertDataset(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def build_ppo_policy(
    *,
    observation_dim: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> PPO:
    """Construct an offline PPO model whose policy can later be fine-tuned."""
    return PPO(
        "MlpPolicy",
        OfflineBalatroEnv(observation_dim),
        learning_rate=learning_rate,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=seed,
        device=device,
        verbose=0,
    )


def _make_splits(
    dataset: ExpertDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = max(1, round(len(dataset) * validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    training_size = len(dataset) - validation_size
    return random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate_policy(
    model: PPO,
    loader: DataLoader,
) -> BCMetrics:
    """Measure negative log likelihood and action prediction accuracy."""
    model.policy.set_training_mode(False)
    device = model.device
    loss_total = 0.0
    field_correct = 0
    field_total = 0
    action_type_correct = 0
    card_correct = 0
    card_total = 0
    exact_correct = 0
    sample_total = 0

    with torch.no_grad():
        for observations, actions in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            _, log_probability, _ = model.policy.evaluate_actions(
                observations,
                actions,
            )
            predicted = model.policy.get_distribution(
                observations
            ).get_actions(deterministic=True)
            batch_size = actions.shape[0]
            loss_total += float((-log_probability).sum().item())
            matches = predicted == actions
            field_correct += int(matches.sum().item())
            field_total += int(matches.numel())
            action_type_correct += int(matches[:, 0].sum().item())
            card_correct += int(matches[:, 1:].sum().item())
            card_total += int(matches[:, 1:].numel())
            exact_correct += int(matches.all(dim=1).sum().item())
            sample_total += batch_size

    return BCMetrics(
        loss=loss_total / sample_total,
        field_accuracy=field_correct / field_total,
        action_type_accuracy=action_type_correct / sample_total,
        card_accuracy=card_correct / card_total,
        exact_action_accuracy=exact_correct / sample_total,
    )


def train_behavioral_cloning(
    *,
    data_path: str | Path,
    model_path: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Train an SB3 PPO policy with supervised expert actions."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = load_expert_data(data_path)
    training_set, validation_set = _make_splits(
        dataset,
        validation_fraction,
        seed,
    )
    effective_batch_size = min(batch_size, len(training_set))
    training_loader = DataLoader(
        training_set,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=min(batch_size, len(validation_set)),
        shuffle=False,
    )

    model = build_ppo_policy(
        observation_dim=dataset.observation_dim,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    optimizer = model.policy.optimizer
    best_validation_loss = float("inf")
    best_epoch = 0
    best_training_metrics: BCMetrics | None = None
    best_validation_metrics: BCMetrics | None = None
    output_path = Path(model_path)
    if output_path.suffix.lower() != ".zip":
        output_path = Path(f"{output_path}.zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        for observations, actions in training_loader:
            observations = observations.to(model.device)
            actions = actions.to(model.device)
            _, log_probability, _ = model.policy.evaluate_actions(
                observations,
                actions,
            )
            loss = -log_probability.mean()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            optimizer.step()

        training_metrics = evaluate_policy(model, training_loader)
        validation_metrics = evaluate_policy(model, validation_loader)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_nll={training_metrics.loss:.4f} "
            f"val_nll={validation_metrics.loss:.4f} "
            f"val_action={validation_metrics.action_type_accuracy:.3f} "
            f"val_cards={validation_metrics.card_accuracy:.3f} "
            f"val_exact={validation_metrics.exact_action_accuracy:.3f}"
        )
        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            best_training_metrics = training_metrics
            best_validation_metrics = validation_metrics
            model.save(output_path)

    assert best_training_metrics is not None
    assert best_validation_metrics is not None
    summary = {
        "data_path": str(Path(data_path).resolve()),
        "model_path": str(output_path.resolve()),
        "records": len(dataset),
        "training_records": len(training_set),
        "validation_records": len(validation_set),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_training_metrics": asdict(best_training_metrics),
        "best_validation_metrics": asdict(best_validation_metrics),
        "final_training_metrics": asdict(training_metrics),
        "final_validation_metrics": asdict(validation_metrics),
        "observation_dimension": dataset.observation_dim,
        "observation_version": observation_version_for_dimension(dataset.observation_dim),
        "action_dimension": ACTION_DIM,
        "seed": seed,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)
    return summary
