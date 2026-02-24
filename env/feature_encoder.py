"""
Encodes raw game state (from bridge state.json) into a normalized feature vector
for the neural network. Handles variable hand size by padding to max hand size.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np

# Card value string -> numeric 2-14 (Ace high)
VALUE_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
SUIT_MAP = {"Spades": 0, "Hearts": 1, "Diamonds": 2, "Clubs": 3}

# Observation layout (all normalized to [0, 1] where applicable):
# [0] phase (0-1, phase/20 cap)
# [1] money (0-1, cap 500)
# [2] chips (0-1, cap 10000)
# [3] blind_chips (0-1, cap 10000)
# [4] hands_left (0-1, max 4)
# [5] discards_left (0-1, max 3)
# [6:6+MAX_HAND*3] for each card: present, value_norm, suit_norm
# [6+MAX_HAND*3:] optional hand_levels: for each of HAND_TYPES_ORDER, (level_norm, chips_norm, mult_norm)
MAX_HAND = 8
HAND_TYPES_ORDER = (
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush",
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


def value_to_norm(val: str | int) -> float:
    if isinstance(val, int):
        v = val
    else:
        v = VALUE_MAP.get(str(val).upper(), 2)
    return (v - VALUE_MIN) / (VALUE_MAX - VALUE_MIN)


def suit_to_norm(suit: str | int) -> float:
    if isinstance(suit, int):
        return suit / 3.0
    return SUIT_MAP.get(str(suit), 0) / 3.0


def encode_state(raw: dict[str, Any]) -> np.ndarray:
    """
    Encode raw state dict (from state.json) into a fixed-size float32 vector.
    """
    phase = min(float(raw.get("phase", 0)) / PHASE_SCALE, 1.0)
    money = min(float(raw.get("money", 0)) / MONEY_CAP, 1.0)
    chips = min(float(raw.get("chips", 0)) / CHIPS_CAP, 1.0)
    blind_chips = min(float(raw.get("blind_chips", 0)) / CHIPS_CAP, 1.0)
    hands_left = min(float(raw.get("hands_left", 0)) / HANDS_MAX, 1.0)
    discards_left = min(float(raw.get("discards_left", 0)) / DISCARDS_MAX, 1.0)

    feats = [phase, money, chips, blind_chips, hands_left, discards_left]
    hand = raw.get("hand") or []
    for i in range(MAX_HAND):
        if i < len(hand):
            c = hand[i]
            val = c.get("value", "2")
            suit = c.get("suit", "Spades")
            feats.append(1.0)
            feats.append(value_to_norm(val))
            feats.append(suit_to_norm(suit))
        else:
            feats.extend([0.0, 0.0, 0.0])
    hand_levels = raw.get("hand_levels") or {}
    for name in HAND_TYPES_ORDER:
        data = hand_levels.get(name)
        if isinstance(data, dict):
            lvl = min(float(data.get("level", 0)) / LEVEL_MAX, 1.0)
            c = min(float(data.get("chips", 0)) / CHIPS_BASE_MAX, 1.0)
            m = min(float(data.get("mult", 0)) / MULT_BASE_MAX, 1.0)
            feats.extend([lvl, c, m])
        else:
            feats.extend([0.0, 0.0, 0.0])
    return np.array(feats, dtype=np.float32)


def load_state_json(path: Path) -> dict[str, Any] | None:
    """Load and parse state.json. Returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
