"""Legal-action enumeration and conservative Balatro hand scoring."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable


MAX_SELECTED_CARDS = 5
HAND_STRENGTH = (
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
RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
RANK_CHIPS = {rank: min(value, 10) for rank, value in RANK_VALUES.items()}
RANK_CHIPS["A"] = 11
SUITS = {"S", "H", "D", "C"}


class ScoringError(ValueError):
    """Raised when a requested action cannot be scored from visible information."""


class HiddenCardError(ScoringError):
    """Raised when classification depends on a face-down card identity."""


@dataclass(frozen=True)
class CandidateAction:
    kind: str
    card_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "card_indices": list(self.card_indices)}


@dataclass(frozen=True)
class HandEvaluation:
    hand_type: str
    selected_indices: tuple[int, ...]
    scoring_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    hand_type: str
    selected_indices: tuple[int, ...]
    scoring_indices: tuple[int, ...]
    chips: float
    mult: float
    score: int
    expected_score: float
    exact: bool
    random: bool
    unsupported_effects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enumerate_legal_actions(state: dict[str, Any]) -> list[CandidateAction]:
    """Enumerate every legal base play/discard set in deterministic order."""
    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    resources = state.get("resources") if isinstance(state.get("resources"), dict) else {}
    hands = resources.get("hands") if isinstance(resources.get("hands"), dict) else {}
    discards = resources.get("discards") if isinstance(resources.get("discards"), dict) else {}
    hands_left = hands.get("remaining", 0)
    discards_left = discards.get("remaining", 0)
    max_cards = min(MAX_SELECTED_CARDS, len(cards))
    selections = [
        selection
        for count in range(1, max_cards + 1)
        for selection in combinations(range(len(cards)), count)
    ]
    actions: list[CandidateAction] = []
    if isinstance(hands_left, (int, float)) and hands_left > 0:
        actions.extend(CandidateAction("play", selection) for selection in selections)
    if isinstance(discards_left, (int, float)) and discards_left > 0:
        actions.extend(CandidateAction("discard", selection) for selection in selections)
    return actions


def _selected_cards(
    state: dict[str, Any], card_indices: Iterable[int]
) -> list[tuple[int, dict[str, Any]]]:
    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    indices = sorted(card_indices)
    if not 1 <= len(indices) <= MAX_SELECTED_CARDS:
        raise ScoringError("A play must select between 1 and 5 cards")
    if len(set(indices)) != len(indices):
        raise ScoringError("Selected card indices contain duplicates")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ScoringError("Selected card indices must be integers")
    if any(index < 0 or index >= len(cards) for index in indices):
        raise ScoringError("Selected card index is outside the hand")
    selected = [(index, cards[index]) for index in indices]
    if any(not isinstance(card, dict) for _, card in selected):
        raise ScoringError("Selected hand entry is not a card object")
    hidden = [index for index, card in selected if card.get("hidden")]
    if hidden:
        raise HiddenCardError(f"Cannot classify face-down selected cards: {hidden}")
    return selected


def _is_stone(card: dict[str, Any]) -> bool:
    return bool(card.get("stone") or "STONE" in card.get("enhancements", []))


def _is_wild(card: dict[str, Any]) -> bool:
    return "WILD" in card.get("enhancements", [])


def _joker_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    jokers = state.get("jokers") if isinstance(state.get("jokers"), dict) else {}
    cards = jokers.get("cards") if isinstance(jokers.get("cards"), list) else []
    return [card for card in cards if isinstance(card, dict)]


def _joker_is_debuffed(joker: dict[str, Any]) -> bool:
    return any(
        str(attribute.get("name", "")).casefold() == "debuff"
        for attribute in joker.get("attributes", [])
        if isinstance(attribute, dict)
    )


def _active_jokers(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [joker for joker in _joker_cards(state) if not joker.get("hidden") and not _joker_is_debuffed(joker)]


def _has_joker(state: dict[str, Any], name: str) -> bool:
    return any(joker.get("name") == name for joker in _active_jokers(state))


def _is_flush(cards: list[dict[str, Any]], *, smeared: bool = False) -> bool:
    if len(cards) != 5 or any(_is_stone(card) for card in cards):
        return False
    if smeared:
        return any(
            all(
                _is_wild(card)
                or (card.get("suit") in {"H", "D"}) == red_suits
                for card in cards
            )
            for red_suits in (True, False)
        )
    return any(
        all(_is_wild(card) or card.get("suit") == suit for card in cards)
        for suit in SUITS
    )


def _is_straight(ranks: list[str]) -> bool:
    if len(ranks) != 5 or len(set(ranks)) != 5:
        return False
    values = sorted(RANK_VALUES[rank] for rank in ranks)
    return values == list(range(values[0], values[0] + 5)) or values == [2, 3, 4, 5, 14]


def classify_hand(state: dict[str, Any], card_indices: Iterable[int]) -> HandEvaluation:
    """Classify a selected play and identify every card that will score."""
    selected = _selected_cards(state, card_indices)
    regular = [(index, card) for index, card in selected if not _is_stone(card)]
    stone_indices = [index for index, card in selected if _is_stone(card)]
    ranks = [str(card.get("rank")) for _, card in regular]
    if any(rank not in RANK_VALUES for rank in ranks):
        raise ScoringError("A visible non-Stone card has an unknown rank")
    rank_groups: dict[str, list[int]] = {}
    for index, card in regular:
        rank_groups.setdefault(str(card["rank"]), []).append(index)
    groups = sorted(
        rank_groups.items(),
        key=lambda item: (len(item[1]), RANK_VALUES[item[0]]),
        reverse=True,
    )
    counts = sorted((len(indices) for indices in rank_groups.values()), reverse=True)
    regular_cards = [card for _, card in regular]
    flush = _is_flush(regular_cards, smeared=_has_joker(state, "Smeared Joker"))
    straight = _is_straight(ranks)
    all_regular_indices = [index for index, _ in regular]

    if len(regular) == 5 and counts == [5] and flush:
        hand_type, pattern_indices = "Flush Five", all_regular_indices
    elif len(regular) == 5 and counts == [3, 2] and flush:
        hand_type, pattern_indices = "Flush House", all_regular_indices
    elif len(regular) == 5 and counts == [5]:
        hand_type, pattern_indices = "Five of a Kind", all_regular_indices
    elif len(regular) == 5 and straight and flush:
        hand_type, pattern_indices = "Straight Flush", all_regular_indices
    elif counts and counts[0] == 4:
        hand_type, pattern_indices = "Four of a Kind", groups[0][1]
    elif len(regular) == 5 and counts == [3, 2]:
        hand_type, pattern_indices = "Full House", all_regular_indices
    elif len(regular) == 5 and flush:
        hand_type, pattern_indices = "Flush", all_regular_indices
    elif len(regular) == 5 and straight:
        hand_type, pattern_indices = "Straight", all_regular_indices
    elif counts and counts[0] == 3:
        hand_type, pattern_indices = "Three of a Kind", groups[0][1]
    elif counts[:2] == [2, 2]:
        pair_groups = [indices for _, indices in groups if len(indices) == 2][:2]
        hand_type, pattern_indices = "Two Pair", [index for group in pair_groups for index in group]
    elif counts and counts[0] == 2:
        hand_type, pattern_indices = "Pair", groups[0][1]
    else:
        hand_type = "High Card"
        pattern_indices = [
            max(regular, key=lambda item: RANK_VALUES[str(item[1]["rank"])])[0]
        ] if regular else []

    scoring_indices = tuple(sorted(set(pattern_indices + stone_indices)))
    return HandEvaluation(
        hand_type=hand_type,
        selected_indices=tuple(index for index, _ in selected),
        scoring_indices=scoring_indices,
    )


def _is_face(card: dict[str, Any], state: dict[str, Any]) -> bool:
    return card.get("rank") in {"J", "Q", "K"} or _has_joker(state, "Pareidolia")


def _card_has_suit(card: dict[str, Any], suit: str, state: dict[str, Any]) -> bool:
    if _is_wild(card):
        return True
    card_suit = card.get("suit")
    if card_suit == suit:
        return True
    if not _has_joker(state, "Smeared Joker"):
        return False
    return {card_suit, suit} <= {"H", "D"} or {card_suit, suit} <= {"S", "C"}


def _rank_counts(cards: list[dict[str, Any]]) -> list[int]:
    counts: dict[str, int] = {}
    for card in cards:
        if _is_stone(card) or card.get("rank") not in RANK_VALUES:
            continue
        rank = str(card["rank"])
        counts[rank] = counts.get(rank, 0) + 1
    return sorted(counts.values(), reverse=True)


def _contains_hand(
    hand_type: str, target: str, selected_cards: list[dict[str, Any]]
) -> bool:
    counts = _rank_counts(selected_cards)
    if target == "Pair":
        return bool(counts and counts[0] >= 2)
    if target == "Three of a Kind":
        return bool(counts and counts[0] >= 3)
    if target == "Two Pair":
        return len([count for count in counts if count >= 2]) >= 2
    if target == "Straight":
        return hand_type in {"Straight", "Straight Flush"}
    if target == "Flush":
        return hand_type in {"Flush", "Straight Flush", "Flush House", "Flush Five"}
    if target == "Four of a Kind":
        return bool(counts and counts[0] >= 4)
    return hand_type == target


def _current_additive(effect: str, unit: str) -> float | None:
    match = re.search(rf"Currently \+([\d.]+)(?: {unit})?", effect, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _current_xmult(effect: str) -> float | None:
    matches = re.findall(r"X([\d.]+)(?: Mult)?", effect, re.IGNORECASE)
    if not matches:
        return None
    current = re.search(r"Currently X([\d.]+)", effect, re.IGNORECASE)
    return float(current.group(1)) if current else float(matches[0])


def _joker_edition(joker: dict[str, Any], chips: float, mult: float) -> tuple[float, float]:
    labels = {
        str(attribute.get("name", "")).upper()
        for attribute in joker.get("attributes", [])
        if isinstance(attribute, dict)
    }
    if "FOIL EDITION" in labels:
        chips += 50
    if "HOLO EDITION" in labels or "HOLOGRAPHIC EDITION" in labels:
        mult += 10
    if "POLYCHROME EDITION" in labels:
        mult *= 1.5
    return chips, mult


def _retrigger_count(
    state: dict[str, Any], evaluation: HandEvaluation, index: int, card: dict[str, Any]
) -> int:
    triggers = 1 + int("RED" in card.get("seals", []))
    jokers = _active_jokers(state)
    if any(joker.get("name") == "Hanging Chad" for joker in jokers) and index == evaluation.scoring_indices[0]:
        triggers += 2
    if any(joker.get("name") == "Sock and Buskin" for joker in jokers) and _is_face(card, state):
        triggers += 1
    if any(joker.get("name") == "Hack" for joker in jokers) and card.get("rank") in {"2", "3", "4", "5"}:
        triggers += 1
    hands_left = state.get("resources", {}).get("hands", {}).get("remaining")
    if any(joker.get("name") == "Dusk" for joker in jokers) and hands_left == 1:
        triggers += 1
    if any(joker.get("name") == "Seltzer" for joker in jokers):
        triggers += 1
    return triggers


def _apply_card_jokers(
    state: dict[str, Any],
    evaluation: HandEvaluation,
    index: int,
    card: dict[str, Any],
    chips: float,
    mult: float,
) -> tuple[float, float, bool, list[str]]:
    random = False
    unsupported: list[str] = []
    first_face = next(
        (
            scoring_index
            for scoring_index in evaluation.scoring_indices
            if _is_face(state["hand"]["cards"][scoring_index], state)
        ),
        None,
    )
    for joker in _active_jokers(state):
        name = str(joker.get("name"))
        effect = str(joker.get("effect") or "")
        if name == "Greedy Joker" and _card_has_suit(card, "D", state):
            mult += 3
        elif name == "Lusty Joker" and _card_has_suit(card, "H", state):
            mult += 3
        elif name == "Wrathful Joker" and _card_has_suit(card, "S", state):
            mult += 3
        elif name == "Gluttonous Joker" and _card_has_suit(card, "C", state):
            mult += 3
        elif name == "Fibonacci" and card.get("rank") in {"A", "2", "3", "5", "8"}:
            mult += 8
        elif name == "Scary Face" and _is_face(card, state):
            chips += 30
        elif name == "Even Steven" and card.get("rank") in {"T", "8", "6", "4", "2"}:
            mult += 4
        elif name == "Odd Todd" and card.get("rank") in {"A", "9", "7", "5", "3"}:
            chips += 31
        elif name == "Scholar" and card.get("rank") == "A":
            chips += 20
            mult += 4
        elif name == "Walkie Talkie" and card.get("rank") in {"T", "4"}:
            chips += 10
            mult += 4
        elif name == "Smiley Face" and _is_face(card, state):
            mult += 5
        elif name == "Photograph" and index == first_face:
            mult *= 2
        elif name == "Arrowhead" and _card_has_suit(card, "S", state):
            chips += 50
        elif name == "Onyx Agate" and _card_has_suit(card, "C", state):
            mult += 7
        elif name == "Ancient Joker":
            suit_names = {"Spade": "S", "Heart": "H", "Diamond": "D", "Club": "C"}
            matched_suit = next((suit for word, suit in suit_names.items() if word in effect), None)
            if matched_suit and _card_has_suit(card, matched_suit, state):
                mult *= 1.5
            elif matched_suit is None:
                unsupported.append("joker:Ancient Joker dynamic suit")
        elif name == "Triboulet" and card.get("rank") in {"K", "Q"}:
            mult *= 2
    return chips, mult, random, unsupported


CARD_TRIGGER_JOKERS = {
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Fibonacci",
    "Scary Face",
    "Even Steven",
    "Odd Todd",
    "Scholar",
    "Walkie Talkie",
    "Smiley Face",
    "Photograph",
    "Arrowhead",
    "Onyx Agate",
    "Ancient Joker",
    "Triboulet",
}
RETRIGGER_JOKERS = {"Hanging Chad", "Sock and Buskin", "Hack", "Dusk", "Seltzer"}
HELD_JOKERS = {"Mime", "Raised Fist", "Baron", "Shoot the Moon"}
STATE_ONLY_JOKERS = {
    "Credit Card",
    "Marble Joker",
    "8 Ball",
    "Chaos the Clown",
    "Delayed Gratification",
    "Space Joker",
    "Egg",
    "Burglar",
    "DNA",
    "Sixth Sense",
    "Faceless Joker",
    "Superposition",
    "To Do List",
    "Riff-Raff",
    "Vagabond",
    "Cloud 9",
    "Rocket",
    "Midas Mask",
    "Luchador",
    "Gift Card",
    "Turtle Bean",
    "Reserved Parking",
    "Mail-In Rebate",
    "To the Moon",
    "Hallucination",
    "Juggler",
    "Drunkard",
    "Golden Joker",
    "Diet Cola",
    "Trading Card",
    "Showman",
    "Merry Andy",
    "Invisible Joker",
    "Satellite",
    "Cartomancer",
    "Astronomer",
    "Burnt Joker",
    "Certificate",
    "Golden Ticket",
    "Business Card",
    "Rough Gem",
    "Matador",
}
CLASSIFICATION_JOKERS = {"Pareidolia", "Splash", "Smeared Joker"}


def _apply_independent_joker(
    joker: dict[str, Any],
    state: dict[str, Any],
    evaluation: HandEvaluation,
    selected_cards: list[dict[str, Any]],
    held_cards: list[dict[str, Any]],
    chips: float,
    mult: float,
) -> tuple[float, float, list[str]]:
    name = str(joker.get("name"))
    effect = str(joker.get("effect") or "")
    unsupported: list[str] = []
    resources = state.get("resources") if isinstance(state.get("resources"), dict) else {}
    discards_left = resources.get("discards", {}).get("remaining", 0)
    hands_left = resources.get("hands", {}).get("remaining", 0)
    money = state.get("money", 0)

    additive_mult_conditions = {
        "Jolly Joker": ("Pair", 8),
        "Zany Joker": ("Three of a Kind", 12),
        "Mad Joker": ("Two Pair", 10),
        "Crazy Joker": ("Straight", 12),
        "Droll Joker": ("Flush", 10),
    }
    additive_chip_conditions = {
        "Sly Joker": ("Pair", 50),
        "Wily Joker": ("Three of a Kind", 100),
        "Clever Joker": ("Two Pair", 80),
        "Devious Joker": ("Straight", 100),
        "Crafty Joker": ("Flush", 80),
    }
    xmult_conditions = {
        "The Duo": ("Pair", 2),
        "The Trio": ("Three of a Kind", 3),
        "The Family": ("Four of a Kind", 4),
        "The Order": ("Straight", 3),
        "The Tribe": ("Flush", 2),
    }
    if name == "Joker":
        mult += 4
    elif name in additive_mult_conditions:
        target, amount = additive_mult_conditions[name]
        if _contains_hand(evaluation.hand_type, target, selected_cards):
            mult += amount
    elif name in additive_chip_conditions:
        target, amount = additive_chip_conditions[name]
        if _contains_hand(evaluation.hand_type, target, selected_cards):
            chips += amount
    elif name in xmult_conditions:
        target, amount = xmult_conditions[name]
        if _contains_hand(evaluation.hand_type, target, selected_cards):
            mult *= amount
    elif name == "Half Joker":
        if len(evaluation.selected_indices) <= 3:
            mult += 20
    elif name == "Joker Stencil":
        joker_state = state.get("jokers", {})
        empty_with_stencil = max(1, int(joker_state.get("limit", 5)) - int(joker_state.get("count", 0)) + 1)
        mult *= empty_with_stencil
    elif name == "Banner":
        chips += 30 * float(discards_left or 0)
    elif name == "Mystic Summit":
        if discards_left == 0:
            mult += 15
    elif name == "Gros Michel":
        mult += 15
    elif name == "Abstract Joker":
        mult += 3 * len(_active_jokers(state))
    elif name == "Blackboard":
        ranked_held = [card for card in held_cards if not _is_stone(card)]
        if ranked_held and all(_card_has_suit(card, "S", state) or _card_has_suit(card, "C", state) for card in ranked_held):
            mult *= 3
    elif name == "Raised Fist":
        ranked_held = [
            card
            for card in held_cards
            if not _is_stone(card) and card.get("rank") in RANK_VALUES
        ]
        if ranked_held:
            lowest = min(ranked_held, key=lambda card: RANK_VALUES[str(card["rank"])])
            mult += 2 * RANK_CHIPS[str(lowest["rank"])]
    elif name == "Hologram":
        value = _current_xmult(effect)
        if value is None:
            unsupported.append("joker:Hologram current value")
        else:
            mult *= value
    elif name == "Constellation":
        value = _current_xmult(effect)
        if value is None:
            unsupported.append("joker:Constellation current value")
        else:
            mult *= value
    elif name == "Cavendish":
        mult *= 3
    elif name == "Card Sharp":
        hand_values = state.get("poker_hands", {}).get(evaluation.hand_type, {})
        if hand_values.get("played_this_round", 0) > 0:
            mult *= 3
    elif name == "Green Joker":
        value = _current_additive(effect, "Mult")
        if value is None:
            unsupported.append("joker:Green Joker current value")
        else:
            mult += value + 1
    elif name == "Red Card":
        value = _current_additive(effect, "Mult")
        if value is None:
            unsupported.append("joker:Red Card current value")
        else:
            mult += value
    elif name == "Fortune Teller":
        value = _current_additive(effect, "Mult")
        if value is None:
            value = _current_additive(effect, "")
        if value is None:
            unsupported.append("joker:Fortune Teller current value")
        else:
            mult += value
    elif name == "Bootstraps":
        mult += 2 * math.floor(float(money or 0) / 5)
    elif name == "Ramen":
        value = _current_xmult(effect)
        if value is None:
            unsupported.append("joker:Ramen current value")
        else:
            mult *= value
    elif name == "Acrobat":
        if hands_left == 1:
            mult *= 3
    elif name == "Supernova":
        hand_values = state.get("poker_hands", {}).get(evaluation.hand_type, {})
        mult += float(hand_values.get("played", 0)) + 1
    elif name == "Runner":
        value = _current_additive(effect, "Chips")
        if value is None:
            unsupported.append("joker:Runner current value")
        else:
            if _contains_hand(evaluation.hand_type, "Straight", selected_cards):
                value += 15
            chips += value
    elif name == "Spare Trousers":
        value = _current_additive(effect, "Mult")
        if value is None:
            unsupported.append("joker:Spare Trousers current value")
        else:
            if _contains_hand(evaluation.hand_type, "Two Pair", selected_cards):
                value += 2
            mult += value
    elif name == "Square Joker":
        value = _current_additive(effect, "Chips")
        if value is None:
            unsupported.append("joker:Square Joker current value")
        else:
            if len(evaluation.selected_indices) == 4:
                value += 4
            chips += value
    elif name == "Swashbuckler":
        value = _current_additive(effect, "Mult")
        if value is None:
            unsupported.append("joker:Swashbuckler current value")
        else:
            mult += value
    elif name == "Stuntman":
        chips += 250
    elif name in CARD_TRIGGER_JOKERS | RETRIGGER_JOKERS | HELD_JOKERS | STATE_ONLY_JOKERS | CLASSIFICATION_JOKERS:
        pass
    else:
        unsupported.append(f"joker:{name}")
    return chips, mult, unsupported


def _apply_scoring_card(
    card: dict[str, Any], chips: float, mult: float
) -> tuple[float, float, bool, list[str]]:
    unsupported: list[str] = []
    random = False
    known_enhancements = {
        "BONUS",
        "MULT",
        "WILD",
        "GLASS",
        "LUCKY",
        "STONE",
        "STEEL",
        "GOLD",
    }
    known_editions = {"FOIL", "HOLO", "HOLOGRAPHIC", "POLYCHROME", "NEGATIVE"}
    known_seals = {"RED", "BLUE", "GOLD", "PURPLE"}
    enhancements = set(card.get("enhancements", []))
    editions = set(card.get("editions", []))
    seals = set(card.get("seals", []))
    unsupported.extend(f"card enhancement:{name}" for name in sorted(enhancements - known_enhancements))
    unsupported.extend(f"card edition:{name}" for name in sorted(editions - known_editions))
    unsupported.extend(f"card seal:{name}" for name in sorted(seals - known_seals))
    chips += 50 if _is_stone(card) else RANK_CHIPS[str(card["rank"])]
    extra_chips = card.get("extra_chips")
    if extra_chips is None:
        unsupported.append("card extra chips:unknown")
    else:
        chips += float(extra_chips)
    if "BONUS" in enhancements:
        chips += 30
    if "MULT" in enhancements:
        mult += 4
    if "LUCKY" in enhancements:
        # Expected +Mult is 20 * 1/5. The separate money chance is not score.
        mult += 4
        random = True
    if "FOIL" in editions:
        chips += 50
    if "HOLO" in editions or "HOLOGRAPHIC" in editions:
        mult += 10
    if "GLASS" in enhancements:
        mult *= 2
        random = True  # The score is deterministic; post-play destruction is not.
    if "POLYCHROME" in editions:
        mult *= 1.5
    return chips, mult, random, unsupported


def score_play(state: dict[str, Any], card_indices: Iterable[int]) -> ScoreResult:
    """Score a base-rule play and mark every unimplemented source of uncertainty."""
    evaluation = classify_hand(state, card_indices)
    poker_hands = (
        state.get("poker_hands") if isinstance(state.get("poker_hands"), dict) else {}
    )
    hand_values = poker_hands.get(evaluation.hand_type)
    if not isinstance(hand_values, dict):
        raise ScoringError(f"Missing level values for {evaluation.hand_type}")
    try:
        chips = float(hand_values["chips"])
        mult = float(hand_values["mult"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoringError(f"Invalid level values for {evaluation.hand_type}") from exc

    hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
    cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    if _has_joker(state, "Splash"):
        evaluation = HandEvaluation(
            hand_type=evaluation.hand_type,
            selected_indices=evaluation.selected_indices,
            scoring_indices=evaluation.selected_indices,
        )
    unsupported: list[str] = []
    random = False
    for index in evaluation.scoring_indices:
        card = cards[index]
        if card.get("debuffed"):
            continue
        for _ in range(_retrigger_count(state, evaluation, index, card)):
            chips, mult, card_random, card_unsupported = _apply_scoring_card(card, chips, mult)
            chips, mult, joker_random, joker_unsupported = _apply_card_jokers(
                state,
                evaluation,
                index,
                card,
                chips,
                mult,
            )
            random = random or card_random or joker_random
            unsupported.extend(card_unsupported)
            unsupported.extend(joker_unsupported)

    selected = set(evaluation.selected_indices)
    held = [
        (index, card)
        for index, card in enumerate(cards)
        if index not in selected and not card.get("hidden") and not card.get("debuffed")
    ]
    active_jokers = _active_jokers(state)
    mime_retrigger = int(any(joker.get("name") == "Mime" for joker in active_jokers))
    # Balatro resolves held-in-hand effects from the right side of the hand.
    for index, card in reversed(held):
        triggers = 1 + int("RED" in card.get("seals", [])) + mime_retrigger
        for _ in range(triggers):
            if "STEEL" in card.get("enhancements", []):
                mult *= 1.5
            for joker in active_jokers:
                name = joker.get("name")
                if name == "Baron" and card.get("rank") == "K":
                    mult *= 1.5
                elif name == "Shoot the Moon" and card.get("rank") == "Q":
                    mult += 13

    selected_cards = [cards[index] for index in evaluation.selected_indices]
    held_cards = [card for _, card in held]
    for joker in active_jokers:
        chips, mult, joker_unsupported = _apply_independent_joker(
            joker,
            state,
            evaluation,
            selected_cards,
            held_cards,
            chips,
            mult,
        )
        unsupported.extend(joker_unsupported)
        chips, mult = _joker_edition(joker, chips, mult)
    for joker in _joker_cards(state):
        if joker.get("hidden"):
            unsupported.append("joker:face down")
    blind = state.get("blind") if isinstance(state.get("blind"), dict) else {}
    if blind.get("effect"):
        unsupported.append(f"blind:{blind.get('name') or blind.get('kind')}")

    expected_score = chips * mult
    score = math.floor(expected_score)
    unique_unsupported = tuple(dict.fromkeys(unsupported))
    return ScoreResult(
        hand_type=evaluation.hand_type,
        selected_indices=evaluation.selected_indices,
        scoring_indices=evaluation.scoring_indices,
        chips=chips,
        mult=mult,
        score=score,
        expected_score=expected_score,
        exact=not unique_unsupported and not random,
        random=random,
        unsupported_effects=unique_unsupported,
    )
