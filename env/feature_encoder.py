"""Versioned encoders for visible Balatro game state."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# V1 is frozen for compatibility with the first recorded dataset and model.
V1_VALUE_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
VALUE_MAP = {
    **V1_VALUE_MAP,
    "JACK": 11,
    "QUEEN": 12,
    "KING": 13,
    "ACE": 14,
}
SUIT_MAP = {"Spades": 0, "Hearts": 1, "Diamonds": 2, "Clubs": 3}

MAX_HAND = 8
HAND_TYPES_ORDER = (
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush",
)
BLIND_KEYS = (
    "bl_small", "bl_big", "bl_ox", "bl_hook", "bl_mouth", "bl_fish",
    "bl_club", "bl_manacle", "bl_tooth", "bl_wall", "bl_house", "bl_mark",
    "bl_final_bell", "bl_wheel", "bl_arm", "bl_psychic", "bl_goad",
    "bl_water", "bl_eye", "bl_plant", "bl_needle", "bl_head",
    "bl_final_leaf", "bl_final_vessel", "bl_window", "bl_serpent",
    "bl_pillar", "bl_flint", "bl_final_acorn", "bl_final_heart",
)
ENHANCEMENT_KEYS = (
    "c_base", "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel",
    "m_stone", "m_gold", "m_lucky",
)
SEAL_KEYS = ("", "Gold", "Red", "Blue", "Purple")
EDITION_KEYS = ("", "foil", "holo", "polychrome", "negative")
CONSUMABLE_SETS = ("Tarot", "Planet", "Spectral", "Voucher")
MAX_JOKERS = 8
MAX_CONSUMABLES = 4
IDENTITY_HASH_BITS = 16
ABILITY_FIELDS = (
    "mult", "chips", "x_mult", "extra", "money", "dollars", "t_mult", "t_chips",
)

LEVEL_MAX = 5.0
CHIPS_BASE_MAX = 100.0
MULT_BASE_MAX = 5.0
PHASE_SCALE = 20.0
MONEY_CAP = 500.0
CHIPS_CAP = 10000.0
HANDS_MAX = 4.0
DISCARDS_MAX = 3.0
VALUE_MIN, VALUE_MAX = 2.0, 14.0

OBS_DIM_V1 = 6 + MAX_HAND * 3 + len(HAND_TYPES_ORDER) * 3
V2_GLOBAL_DIM = 18
V2_BLIND_DIM = len(BLIND_KEYS) + 1
V2_DEBUFF_DIM = len(SUIT_MAP) + 13 + 1 + len(HAND_TYPES_ORDER) + 2
V2_MOST_PLAYED_DIM = len(HAND_TYPES_ORDER) + 1
V2_HAND_SUMMARY_DIM = 13 + len(SUIT_MAP) + 10
V2_CARD_DIM = 1 + 13 + len(SUIT_MAP) + 4 + (len(ENHANCEMENT_KEYS) + 1) + (len(SEAL_KEYS) + 1) + (len(EDITION_KEYS) + 1)
V2_INVENTORY_DIM = 1 + IDENTITY_HASH_BITS + (len(EDITION_KEYS) + 1) + 1 + 1 + len(ABILITY_FIELDS)
V2_CONSUMABLE_DIM = V2_INVENTORY_DIM + len(CONSUMABLE_SETS) + 1
OBS_DIM_V2 = (
    V2_GLOBAL_DIM
    + V2_BLIND_DIM
    + V2_DEBUFF_DIM
    + V2_MOST_PLAYED_DIM
    + V2_HAND_SUMMARY_DIM
    + MAX_HAND * V2_CARD_DIM
    + len(HAND_TYPES_ORDER) * 3
    + MAX_JOKERS * V2_INVENTORY_DIM
    + MAX_CONSUMABLES * V2_CONSUMABLE_DIM
)
OBSERVATION_DIMS = {1: OBS_DIM_V1, 2: OBS_DIM_V2}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _one_hot(value: Any, choices: Iterable[Any]) -> list[float]:
    choices = tuple(choices)
    result = [0.0] * (len(choices) + 1)
    try:
        result[choices.index(value)] = 1.0
    except ValueError:
        result[-1] = 1.0
    return result


def _rank(value: Any, *, legacy: bool = False) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(2, min(int(value), 14))
    mapping = V1_VALUE_MAP if legacy else VALUE_MAP
    return mapping.get(str(value).strip().upper(), 2)


def value_to_norm(val: str | int) -> float:
    """Legacy rank normalization retained for v1 compatibility."""
    return (_rank(val, legacy=True) - VALUE_MIN) / (VALUE_MAX - VALUE_MIN)


def suit_to_norm(suit: str | int) -> float:
    if isinstance(suit, int):
        return suit / 3.0
    return SUIT_MAP.get(str(suit), 0) / 3.0


def _encode_hand_levels(raw: dict[str, Any]) -> list[float]:
    features: list[float] = []
    hand_levels = raw.get("hand_levels") or {}
    for name in HAND_TYPES_ORDER:
        data = hand_levels.get(name)
        if isinstance(data, dict):
            features.extend(
                [
                    _clamp01(_number(data.get("level")) / LEVEL_MAX),
                    _clamp01(_number(data.get("chips")) / CHIPS_BASE_MAX),
                    _clamp01(_number(data.get("mult")) / MULT_BASE_MAX),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])
    return features


def encode_state_v1(raw: dict[str, Any]) -> np.ndarray:
    """Encode the original 60-feature observation without semantic changes."""
    features = [
        min(_number(raw.get("phase")) / PHASE_SCALE, 1.0),
        min(_number(raw.get("money")) / MONEY_CAP, 1.0),
        min(_number(raw.get("chips")) / CHIPS_CAP, 1.0),
        min(_number(raw.get("blind_chips")) / CHIPS_CAP, 1.0),
        min(_number(raw.get("hands_left")) / HANDS_MAX, 1.0),
        min(_number(raw.get("discards_left")) / DISCARDS_MAX, 1.0),
    ]
    hand = raw.get("hand") or []
    for index in range(MAX_HAND):
        if index < len(hand):
            card = hand[index]
            features.extend(
                [
                    1.0,
                    value_to_norm(card.get("value", "2")),
                    suit_to_norm(card.get("suit", "Spades")),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])
    features.extend(_encode_hand_levels(raw))
    return np.asarray(features, dtype=np.float32)


def _encode_globals(raw: dict[str, Any]) -> list[float]:
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    blind = raw.get("blind") if isinstance(raw.get("blind"), dict) else {}
    hand = raw.get("hand") or []
    chips = max(_number(raw.get("chips")), 0.0)
    target = max(_number(raw.get("blind_chips")), 0.0)
    deck_total = max(_number(run.get("deck_total")), 0.0)
    deck_remaining = max(_number(run.get("deck_remaining")), 0.0)
    money = _number(raw.get("money"))
    progress = chips / target if target > 0 else 0.0
    remaining = max(target - chips, 0.0) / target if target > 0 else 0.0
    return [
        _clamp01(_number(raw.get("phase")) / PHASE_SCALE),
        _clamp01((money + 20.0) / 520.0),
        _clamp01(progress),
        _clamp01(remaining),
        _clamp01(math.log1p(chips) / math.log1p(1_000_000_000.0)),
        _clamp01(math.log1p(target) / math.log1p(1_000_000_000.0)),
        _clamp01(_number(raw.get("hands_left")) / 10.0),
        _clamp01(_number(raw.get("discards_left")) / 10.0),
        _clamp01(_number(run.get("ante")) / 39.0),
        _clamp01(_number(run.get("round")) / 100.0),
        _clamp01(_number(run.get("stake")) / 8.0),
        _clamp01(deck_remaining / deck_total) if deck_total else 0.0,
        _clamp01(deck_total / 100.0),
        _clamp01(len(hand) / MAX_HAND),
        _clamp01(_number(run.get("hands_played")) / 10.0),
        _clamp01(_number(run.get("discards_used")) / 10.0),
        float(bool(blind.get("boss"))),
        float(bool(blind.get("disabled"))),
    ]


def _encode_blind(raw: dict[str, Any]) -> list[float]:
    blind = raw.get("blind") if isinstance(raw.get("blind"), dict) else {}
    debuff = blind.get("debuff") if isinstance(blind.get("debuff"), dict) else {}
    features = _one_hot(str(blind.get("key") or ""), BLIND_KEYS)
    debuffed_suit = str(debuff.get("suit") or "")
    features.extend(float(debuffed_suit == suit) for suit in SUIT_MAP)
    debuffed_rank = debuff.get("value") or debuff.get("rank")
    rank_value = _rank(debuffed_rank) if debuffed_rank not in (None, "") else 0
    features.extend(float(rank_value == rank) for rank in range(2, 15))
    features.append(float(bool(debuff.get("is_face"))))
    debuffed_hand = str(debuff.get("hand") or "")
    features.extend(float(debuffed_hand == name) for name in HAND_TYPES_ORDER)
    features.extend(
        [
            _clamp01(_number(debuff.get("h_size_ge")) / MAX_HAND),
            _clamp01(_number(debuff.get("h_size_le")) / MAX_HAND),
        ]
    )
    return features


def _has_straight(ranks: set[int]) -> bool:
    values = set(ranks)
    if 14 in values:
        values.add(1)
    ordered = sorted(values)
    longest = current = 0
    previous: int | None = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, current)
        previous = value
    return longest >= 5


def _encode_hand_summary(hand: list[dict[str, Any]]) -> list[float]:
    rank_counts = {rank: 0 for rank in range(2, 15)}
    suit_counts = {suit: 0 for suit in SUIT_MAP}
    for card in hand:
        facing = str(card.get("facing") or "front").lower()
        if facing not in ("", "front"):
            continue
        rank_counts[_rank(card.get("value"))] += 1
        suit = str(card.get("suit") or "")
        if suit in suit_counts:
            suit_counts[suit] += 1
    counts = list(rank_counts.values())
    pair_count = sum(count >= 2 for count in counts)
    triple_count = sum(count >= 3 for count in counts)
    quad_count = sum(count >= 4 for count in counts)
    maximum_rank = max(counts, default=0)
    maximum_suit = max(suit_counts.values(), default=0)
    features = [_clamp01(count / MAX_HAND) for count in counts]
    features.extend(_clamp01(suit_counts[suit] / MAX_HAND) for suit in SUIT_MAP)
    features.extend(
        [
            _clamp01(sum(count > 0 for count in counts) / 13.0),
            _clamp01(maximum_rank / MAX_HAND),
            _clamp01(maximum_suit / MAX_HAND),
            _clamp01(pair_count / 4.0),
            _clamp01(triple_count / 2.0),
            float(quad_count > 0),
            float(pair_count >= 2),
            float(triple_count >= 1 and pair_count >= 2),
            float(_has_straight({rank for rank, count in rank_counts.items() if count})),
            float(maximum_suit >= 5),
        ]
    )
    return features


def _encode_card(card: dict[str, Any] | None) -> list[float]:
    if card is None:
        return [0.0] * V2_CARD_DIM
    rank = _rank(card.get("value"))
    suit = str(card.get("suit") or "")
    facing = str(card.get("facing") or "front").lower()
    face_down = facing not in ("", "front")
    if face_down:
        features = [1.0] + [0.0] * 13 + [0.0] * len(SUIT_MAP)
        features.extend(
            [
                0.0,
                1.0,
                float(bool(card.get("forced"))),
                0.0,
            ]
        )
        features.extend([0.0] * (len(ENHANCEMENT_KEYS) + 1))
        features.extend([0.0] * (len(SEAL_KEYS) + 1))
        features.extend([0.0] * (len(EDITION_KEYS) + 1))
        return features
    features = [1.0]
    features.extend(float(rank == candidate) for candidate in range(2, 15))
    features.extend(float(suit == candidate) for candidate in SUIT_MAP)
    features.extend(
        [
            float(bool(card.get("debuff"))),
            0.0,
            float(bool(card.get("forced"))),
            float(bool(card.get("played_this_ante"))),
        ]
    )
    features.extend(_one_hot(str(card.get("center") or ""), ENHANCEMENT_KEYS))
    features.extend(_one_hot(str(card.get("seal") or ""), SEAL_KEYS))
    features.extend(_one_hot(str(card.get("edition") or "").lower(), EDITION_KEYS))
    return features


def _identity_bits(value: Any) -> list[float]:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).digest()
    return [float((digest[index // 8] >> (index % 8)) & 1) for index in range(IDENTITY_HASH_BITS)]


def _find_numeric(ability: Any, target: str) -> float:
    if not isinstance(ability, dict):
        return 0.0
    target = target.lower()
    stack = [ability]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            normalized_key = str(key).lower().replace("xmult", "x_mult")
            if normalized_key == target and isinstance(value, (int, float)) and not isinstance(value, bool):
                return _number(value)
            if isinstance(value, dict):
                stack.append(value)
    return 0.0


def _encode_inventory_card(card: dict[str, Any] | None, *, consumable: bool) -> list[float]:
    expected = V2_CONSUMABLE_DIM if consumable else V2_INVENTORY_DIM
    if card is None:
        return [0.0] * expected
    features = [1.0]
    features.extend(_identity_bits(card.get("key") or card.get("name")))
    features.extend(_one_hot(str(card.get("edition") or "").lower(), EDITION_KEYS))
    features.extend(
        [
            float(bool(card.get("debuff"))),
            _clamp01(_number(card.get("sell_cost")) / 50.0),
        ]
    )
    ability = card.get("ability") if isinstance(card.get("ability"), dict) else {}
    for field in ABILITY_FIELDS:
        value = _find_numeric(ability, field)
        features.append(_clamp01(0.5 + max(-1000.0, min(value, 1000.0)) / 2000.0))
    if consumable:
        features.extend(_one_hot(str(card.get("set") or ""), CONSUMABLE_SETS))
    return features


def encode_state_v2(raw: dict[str, Any]) -> np.ndarray:
    """Encode visible gameplay information. The run seed is metadata, not a feature."""
    features = _encode_globals(raw)
    features.extend(_encode_blind(raw))
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    features.extend(_one_hot(str(run.get("most_played_hand") or ""), HAND_TYPES_ORDER))
    hand = raw.get("hand") or []
    features.extend(_encode_hand_summary(hand))
    for index in range(MAX_HAND):
        features.extend(_encode_card(hand[index] if index < len(hand) else None))
    features.extend(_encode_hand_levels(raw))
    jokers = raw.get("jokers") or []
    for index in range(MAX_JOKERS):
        features.extend(_encode_inventory_card(jokers[index] if index < len(jokers) else None, consumable=False))
    consumables = raw.get("consumables") or []
    for index in range(MAX_CONSUMABLES):
        features.extend(_encode_inventory_card(consumables[index] if index < len(consumables) else None, consumable=True))
    if len(features) != OBS_DIM_V2:
        raise RuntimeError(f"v2 encoder produced {len(features)} features; expected {OBS_DIM_V2}")
    return np.asarray(features, dtype=np.float32)


def encode_state(raw: dict[str, Any], observation_version: int = 1) -> np.ndarray:
    """Encode state with the requested observation contract."""
    if observation_version == 1:
        return encode_state_v1(raw)
    if observation_version == 2:
        return encode_state_v2(raw)
    raise ValueError(f"Unsupported observation version: {observation_version}")


def observation_dimension(observation_version: int) -> int:
    try:
        return OBSERVATION_DIMS[observation_version]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation version: {observation_version}") from exc


def observation_version_for_dimension(dimension: int) -> int:
    for version, expected in OBSERVATION_DIMS.items():
        if dimension == expected:
            return version
    raise ValueError(
        f"Model uses unsupported observation dimension {dimension}; "
        f"expected one of {sorted(OBSERVATION_DIMS.values())}"
    )


def load_state_json(path: Path) -> dict[str, Any] | None:
    """Load state.json, returning None while a bridge write is incomplete."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as state_file:
            return json.load(state_file)
    except (json.JSONDecodeError, OSError):
        return None
