"""Fixed-vocabulary policy baseline for prepared Balatro planner states."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


MAX_HAND_SLOTS = 12
MAX_JOKER_SLOTS = 8
MAX_CONSUMABLE_SLOTS = 4
HASH_BITS = 24
HAND_TYPES = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("S", "H", "D", "C")
ENHANCEMENTS = ("BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY")
EDITIONS = ("FOIL", "HOLO", "HOLOGRAPHIC", "POLYCHROME", "NEGATIVE")
SEALS = ("RED", "BLUE", "GOLD", "PURPLE")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            yield value


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(low, min(number, high))


def _norm(value: Any, scale: float) -> float:
    return _clamp(value, 0.0, scale) / scale


def _signed_norm(value: Any, scale: float) -> float:
    return 0.5 + _clamp(value, -scale, scale) / (2.0 * scale)


def _one_hot(value: Any, choices: tuple[Any, ...]) -> list[float]:
    return [float(value == choice) for choice in choices]


def _hash_bits(value: Any, bits: int = HASH_BITS) -> list[float]:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).digest()
    return [float((digest[index // 8] >> (index % 8)) & 1) for index in range(bits)]


def _dynamic_effect_values(effect: str) -> list[float]:
    current_add = re.search(r"Currently \+([\d.]+)(?: (Chips|Mult))?", effect, re.IGNORECASE)
    current_x = re.search(r"Currently X([\d.]+)", effect, re.IGNORECASE)
    first_chip = re.search(r"\+([\d.]+) Chips", effect, re.IGNORECASE)
    first_mult = re.search(r"\+([\d.]+) Mult", effect, re.IGNORECASE)
    first_x = re.search(r"X([\d.]+)(?: Mult)?", effect, re.IGNORECASE)
    additive = float(current_add.group(1)) if current_add else 0.0
    additive_unit = current_add.group(2).casefold() if current_add and current_add.group(2) else ""
    return [
        _signed_norm(additive if additive_unit == "chips" else 0.0, 1000.0),
        _signed_norm(additive if additive_unit in {"", "mult"} else 0.0, 200.0),
        _norm(float(current_x.group(1)) if current_x else 0.0, 20.0),
        _signed_norm(float(first_chip.group(1)) if first_chip else 0.0, 1000.0),
        _signed_norm(float(first_mult.group(1)) if first_mult else 0.0, 200.0),
        _norm(float(first_x.group(1)) if first_x else 0.0, 20.0),
    ]


def _encode_card(card: dict[str, Any] | None) -> list[float]:
    if card is None:
        return [0.0] * card_feature_dimension()
    hidden = bool(card.get("hidden"))
    features = [1.0, float(hidden), float(bool(card.get("stone")))]
    features.extend([0.0] * len(RANKS) if hidden else _one_hot(card.get("rank"), RANKS))
    features.extend([0.0] * len(SUITS) if hidden else _one_hot(card.get("suit"), SUITS))
    enhancements = set(card.get("enhancements", []))
    editions = set(card.get("editions", []))
    seals = set(card.get("seals", []))
    features.extend(float(name in enhancements) for name in ENHANCEMENTS)
    features.extend(float(name in editions) for name in EDITIONS)
    features.extend(float(name in seals) for name in SEALS)
    features.append(float(bool(card.get("debuffed"))))
    extra_chips = card.get("extra_chips")
    features.extend([float(extra_chips is None), _norm(extra_chips or 0, 500.0)])
    return features


def card_feature_dimension() -> int:
    return 3 + len(RANKS) + len(SUITS) + len(ENHANCEMENTS) + len(EDITIONS) + len(SEALS) + 3


def _encode_joker(joker: dict[str, Any] | None) -> list[float]:
    if joker is None:
        return [0.0] * joker_feature_dimension()
    features = [1.0, float(bool(joker.get("hidden")))]
    features.extend(_hash_bits(joker.get("name")))
    features.extend(_hash_bits(joker.get("effect")))
    features.append(_norm(joker.get("sell_value", 0), 50.0))
    features.extend(_dynamic_effect_values(str(joker.get("effect") or "")))
    attributes = {
        str(item.get("name", "")).upper()
        for item in joker.get("attributes", [])
        if isinstance(item, dict)
    }
    features.extend(
        float(f"{edition} EDITION" in attributes)
        for edition in ("FOIL", "HOLO", "HOLOGRAPHIC", "POLYCHROME", "NEGATIVE")
    )
    return features


def joker_feature_dimension() -> int:
    return 2 + HASH_BITS * 2 + 1 + 6 + 5


def _encode_consumable(card: dict[str, Any] | None) -> list[float]:
    if card is None:
        return [0.0] * consumable_feature_dimension()
    features = [1.0, float(bool(card.get("hidden")))]
    features.extend(_hash_bits(card.get("name")))
    features.extend(_hash_bits(card.get("effect")))
    features.append(_norm(card.get("sell_value", 0), 50.0))
    return features


def consumable_feature_dimension() -> int:
    return 2 + HASH_BITS * 2 + 1


def encode_planner_state(state: dict[str, Any]) -> np.ndarray:
    """Encode only policy-visible state; provenance, outcomes, and seeds are absent."""
    resources = state.get("resources") if isinstance(state.get("resources"), dict) else {}
    hands = resources.get("hands") if isinstance(resources.get("hands"), dict) else {}
    discards = resources.get("discards") if isinstance(resources.get("discards"), dict) else {}
    blind = state.get("blind") if isinstance(state.get("blind"), dict) else {}
    ante = state.get("ante") if isinstance(state.get("ante"), dict) else {}
    current_score = max(float(blind.get("current_score") or 0), 0.0)
    target_score = max(float(blind.get("target_score") or 0), 0.0)
    features = [
        _norm(state.get("round"), 100.0),
        _norm(ante.get("current"), 39.0),
        _norm(ante.get("max"), 39.0),
        _signed_norm(state.get("money"), 500.0),
        _norm(hands.get("remaining"), 12.0),
        _norm(hands.get("max"), 12.0),
        _norm(discards.get("remaining"), 12.0),
        _norm(discards.get("max"), 12.0),
        min(current_score / target_score, 2.0) / 2.0 if target_score else 0.0,
        math.log1p(current_score) / math.log1p(1_000_000_000.0),
        math.log1p(target_score) / math.log1p(1_000_000_000.0),
    ]
    features.extend(_one_hot(blind.get("kind"), ("SMALL", "BIG", "BOSS")))
    features.extend(_hash_bits(blind.get("name"), 16))
    features.extend(_hash_bits(blind.get("effect"), 16))
    features.extend(_hash_bits(state.get("deck"), 8))
    features.extend(_hash_bits(state.get("stake"), 8))

    poker_hands = state.get("poker_hands") if isinstance(state.get("poker_hands"), dict) else {}
    for name in HAND_TYPES:
        values = poker_hands.get(name) if isinstance(poker_hands.get(name), dict) else {}
        features.extend(
            [
                _norm(values.get("level"), 20.0),
                _norm(values.get("chips"), 1000.0),
                _norm(values.get("mult"), 100.0),
                _norm(values.get("played"), 500.0),
                _norm(values.get("played_this_round"), 20.0),
            ]
        )

    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    hand_cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    if len(hand_cards) > MAX_HAND_SLOTS:
        raise ValueError(f"Hand has {len(hand_cards)} cards; maximum is {MAX_HAND_SLOTS}")
    features.extend([_norm(len(hand_cards), MAX_HAND_SLOTS), _norm(hand.get("limit"), MAX_HAND_SLOTS)])
    for index in range(MAX_HAND_SLOTS):
        features.extend(_encode_card(hand_cards[index] if index < len(hand_cards) else None))

    jokers = state.get("jokers") if isinstance(state.get("jokers"), dict) else {}
    joker_cards = jokers.get("cards") if isinstance(jokers.get("cards"), list) else []
    features.extend([_norm(len(joker_cards), MAX_JOKER_SLOTS), _norm(jokers.get("limit"), MAX_JOKER_SLOTS)])
    for index in range(MAX_JOKER_SLOTS):
        features.extend(_encode_joker(joker_cards[index] if index < len(joker_cards) else None))

    consumables = state.get("consumables") if isinstance(state.get("consumables"), dict) else {}
    consumable_cards = (
        consumables.get("cards") if isinstance(consumables.get("cards"), list) else []
    )
    features.extend(
        [
            _norm(len(consumable_cards), MAX_CONSUMABLE_SLOTS),
            _norm(consumables.get("limit"), MAX_CONSUMABLE_SLOTS),
        ]
    )
    for index in range(MAX_CONSUMABLE_SLOTS):
        features.extend(
            _encode_consumable(consumable_cards[index] if index < len(consumable_cards) else None)
        )
    array = np.asarray(features, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Planner state encoder produced non-finite values")
    return array


def build_action_vocabulary(max_hand_slots: int = MAX_HAND_SLOTS) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (kind, indices)
        for kind in ("play", "discard")
        for count in range(1, 6)
        for indices in combinations(range(max_hand_slots), count)
    )


ACTION_VOCABULARY = build_action_vocabulary()
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_VOCABULARY)}


def legal_action_mask(state: dict[str, Any]) -> np.ndarray:
    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    hand_size = len(hand.get("cards")) if isinstance(hand.get("cards"), list) else 0
    resources = state.get("resources") if isinstance(state.get("resources"), dict) else {}
    hands_left = resources.get("hands", {}).get("remaining", 0)
    discards_left = resources.get("discards", {}).get("remaining", 0)
    return np.asarray(
        [
            max(indices, default=-1) < hand_size
            and ((kind == "play" and hands_left > 0) or (kind == "discard" and discards_left > 0))
            for kind, indices in ACTION_VOCABULARY
        ],
        dtype=np.bool_,
    )


class PreparedPlannerDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        observations: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        legal_masks: list[np.ndarray] = []
        consensus: list[int] = []
        for record in _iter_jsonl(self.path):
            state = record.get("state")
            action_targets = record.get("action_targets")
            if not isinstance(state, dict) or not isinstance(action_targets, list):
                raise ValueError(f"{self.path}: record is missing state/action_targets")
            target = np.zeros(len(ACTION_VOCABULARY), dtype=np.float32)
            for action in action_targets:
                key = (str(action.get("kind")), tuple(sorted(action.get("card_indices", []))))
                try:
                    target[ACTION_TO_INDEX[key]] += float(action.get("target_probability", 0))
                except KeyError as exc:
                    raise ValueError(f"{self.path}: target action is outside vocabulary: {key}") from exc
            if not np.isclose(target.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{self.path}: action probabilities sum to {target.sum()}")
            legal = legal_action_mask(state)
            if np.any((target > 0) & ~legal):
                raise ValueError(f"{self.path}: target distribution includes an illegal action")
            observations.append(encode_planner_state(state))
            targets.append(target)
            legal_masks.append(legal)
            consensus.append(int(target.argmax()))
        if not observations:
            raise ValueError(f"Prepared dataset is empty: {self.path}")
        self.observations = torch.as_tensor(np.stack(observations), dtype=torch.float32)
        self.targets = torch.as_tensor(np.stack(targets), dtype=torch.float32)
        self.legal_masks = torch.as_tensor(np.stack(legal_masks), dtype=torch.bool)
        self.consensus = torch.as_tensor(consensus, dtype=torch.long)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int):
        return (
            self.observations[index],
            self.targets[index],
            self.legal_masks[index],
            self.consensus[index],
        )

    @property
    def observation_dimension(self) -> int:
        return int(self.observations.shape[1])


class PlannerPolicy(nn.Module):
    def __init__(self, observation_dimension: int, action_count: int = len(ACTION_VOCABULARY)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dimension, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


@dataclass
class PlannerMetrics:
    nll: float
    consensus_accuracy: float
    expected_teacher_agreement: float
    action_kind_accuracy: float
    top5_target_coverage: float


def _masked_logits(model: PlannerPolicy, observations: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return model(observations).masked_fill(~legal, -1e9)


def evaluate_planner_policy(
    model: PlannerPolicy, loader: DataLoader, device: torch.device
) -> PlannerMetrics:
    model.eval()
    total_loss = 0.0
    consensus_correct = 0
    expected_agreement = 0.0
    kind_correct = 0
    top5_coverage = 0
    records = 0
    action_kinds = torch.as_tensor(
        [0 if kind == "play" else 1 for kind, _ in ACTION_VOCABULARY],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        for observations, targets, legal, consensus in loader:
            observations = observations.to(device)
            targets = targets.to(device)
            legal = legal.to(device)
            consensus = consensus.to(device)
            logits = _masked_logits(model, observations, legal)
            log_probabilities = torch.log_softmax(logits, dim=1)
            loss = -(targets * log_probabilities).sum(dim=1)
            predicted = logits.argmax(dim=1)
            top5 = logits.topk(k=5, dim=1).indices
            total_loss += float(loss.sum().item())
            consensus_correct += int((predicted == consensus).sum().item())
            expected_agreement += float(targets.gather(1, predicted[:, None]).sum().item())
            kind_correct += int((action_kinds[predicted] == action_kinds[consensus]).sum().item())
            top5_coverage += int((targets.gather(1, top5) > 0).any(dim=1).sum().item())
            records += observations.shape[0]
    return PlannerMetrics(
        nll=total_loss / records,
        consensus_accuracy=consensus_correct / records,
        expected_teacher_agreement=expected_agreement / records,
        action_kind_accuracy=kind_correct / records,
        top5_target_coverage=top5_coverage / records,
    )


def train_planner_policy(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    model_path: str | Path,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    training = PreparedPlannerDataset(train_path)
    validation = PreparedPlannerDataset(validation_path)
    if training.observation_dimension != validation.observation_dimension:
        raise ValueError("Training and validation observation dimensions differ")
    train_loader = DataLoader(
        training,
        batch_size=min(batch_size, len(training)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=min(batch_size, len(validation)),
        shuffle=False,
    )
    model = PlannerPolicy(training.observation_dimension).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    output = Path(model_path)
    if output.suffix.lower() != ".pt":
        output = output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    best_metrics: PlannerMetrics | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for observations, targets, legal, _ in train_loader:
            observations = observations.to(selected_device)
            targets = targets.to(selected_device)
            legal = legal.to(selected_device)
            logits = _masked_logits(model, observations, legal)
            loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        training_metrics = evaluate_planner_policy(model, train_loader, selected_device)
        validation_metrics = evaluate_planner_policy(model, validation_loader, selected_device)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_nll={training_metrics.nll:.4f} val_nll={validation_metrics.nll:.4f} "
            f"val_exact={validation_metrics.consensus_accuracy:.3f} "
            f"val_agree={validation_metrics.expected_teacher_agreement:.3f} "
            f"val_kind={validation_metrics.action_kind_accuracy:.3f} "
            f"val_top5={validation_metrics.top5_target_coverage:.3f}"
        )
        if validation_metrics.nll < best_validation:
            best_validation = validation_metrics.nll
            best_epoch = epoch
            best_metrics = validation_metrics
            torch.save(
                {
                    "schema_version": 1,
                    "label_source": "aggregated_unrated_llm_votes",
                    "model_state_dict": model.state_dict(),
                    "observation_dimension": training.observation_dimension,
                    "action_count": len(ACTION_VOCABULARY),
                    "max_hand_slots": MAX_HAND_SLOTS,
                    "action_vocabulary": [
                        {"kind": kind, "card_indices": list(indices)}
                        for kind, indices in ACTION_VOCABULARY
                    ],
                    "best_epoch": best_epoch,
                    "validation_metrics": asdict(best_metrics),
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                    },
                    "seed": seed,
                },
                output,
            )
    assert best_metrics is not None
    summary = {
        "schema_version": 1,
        "label_source": "aggregated_unrated_llm_votes",
        "train_path": str(Path(train_path).resolve()),
        "validation_path": str(Path(validation_path).resolve()),
        "model_path": str(output.resolve()),
        "device": str(selected_device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "training_records": len(training),
        "validation_records": len(validation),
        "observation_dimension": training.observation_dimension,
        "action_count": len(ACTION_VOCABULARY),
        "best_validation_metrics": asdict(best_metrics),
        "seed": seed,
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
