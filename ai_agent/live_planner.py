"""Adapt live bridge states and run planner-policy inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .planner_policy import (
    ACTION_VOCABULARY,
    HAND_TYPES,
    MAX_HAND_SLOTS,
    PlannerPolicy,
    encode_planner_state,
    legal_action_mask,
)
from .scoring_engine import ScoringError, score_play


RANK_NAMES = {
    "10": "T",
    "JACK": "J",
    "QUEEN": "Q",
    "KING": "K",
    "ACE": "A",
}
SUIT_NAMES = {"SPADES": "S", "HEARTS": "H", "DIAMONDS": "D", "CLUBS": "C"}
STAKE_NAMES = {
    1: "WHITE",
    2: "RED",
    3: "GREEN",
    4: "BLACK",
    5: "BLUE",
    6: "PURPLE",
    7: "ORANGE",
    8: "GOLD",
}
CENTER_ENHANCEMENTS = {
    "m_bonus": "BONUS",
    "m_mult": "MULT",
    "m_wild": "WILD",
    "m_glass": "GLASS",
    "m_steel": "STEEL",
    "m_stone": "STONE",
    "m_gold": "GOLD",
    "m_lucky": "LUCKY",
}
EDITION_NAMES = {
    "foil": "FOIL",
    "holo": "HOLOGRAPHIC",
    "polychrome": "POLYCHROME",
    "negative": "NEGATIVE",
}
BLIND_EFFECTS = {
    "Amber Acorn": "Flips and shuffles all Joker cards",
    "Crimson Heart": "One random Joker disabled every hand",
    "The Arm": "Decrease level of played poker hand",
    "The Club": "All Club cards are debuffed",
    "The Fish": "Cards drawn face down after each hand played",
    "The Flint": "Base Chips and Mult are halved",
    "The Goad": "All Spade cards are debuffed",
    "The Head": "All Heart cards are debuffed",
    "The Hook": "Discards 2 random cards per hand played",
    "The House": "First hand is drawn face down",
    "The Manacle": "-1 Hand Size",
    "The Mark": "All face cards are drawn face down",
    "The Mouth": "Play only 1 hand type this round",
    "The Needle": "Play only 1 hand",
    "The Pillar": "Cards played previously this Ante are debuffed",
    "The Plant": "All face cards are debuffed",
    "The Psychic": "Must play 5 cards",
    "The Water": "Start with 0 discards",
    "The Wheel": "#1# in #2# cards get drawn face down",
    "The Window": "All Diamond cards are debuffed",
}


def _rank_name(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return None
    return RANK_NAMES.get(normalized, normalized)


def _suit_name(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return None
    return SUIT_NAMES.get(normalized, normalized[:1])


def _blind_kind(blind: dict[str, Any]) -> str:
    key = str(blind.get("key") or "").casefold()
    if key == "bl_small":
        return "SMALL"
    if key == "bl_big":
        return "BIG"
    blind_type = str(blind.get("type") or "").upper()
    return "BOSS" if blind.get("boss") or blind_type == "BOSS" else blind_type


def _live_card(card: dict[str, Any], index: int) -> dict[str, Any]:
    center = str(card.get("center") or "").casefold()
    facing = str(card.get("facing") or "front").casefold()
    hidden = facing not in {"", "front"}
    enhancement = CENTER_ENHANCEMENTS.get(center)
    edition = EDITION_NAMES.get(str(card.get("edition") or "").casefold())
    seal = str(card.get("seal") or "").strip().upper()
    stone = center == "m_stone"
    return {
        "index": index,
        "hidden": hidden,
        "stone": stone,
        "rank": None if hidden or stone else _rank_name(card.get("value")),
        "suit": None if hidden or stone else _suit_name(card.get("suit")),
        "enhancements": [enhancement] if enhancement and not stone else [],
        "editions": [edition] if edition else [],
        "seals": [seal] if seal else [],
        "debuffed": bool(card.get("debuff")),
        "extra_chips": float(card.get("extra_chips") or 0),
    }


def _live_inventory_card(card: dict[str, Any], index: int) -> dict[str, Any]:
    edition = EDITION_NAMES.get(str(card.get("edition") or "").casefold())
    attributes = []
    if edition:
        attributes.append({"name": f"{edition.title()} Edition", "value": ""})
    return {
        "index": index,
        "name": str(card.get("name") or card.get("key") or ""),
        "effect": "",
        "hidden": False,
        "attributes": attributes,
        "sell_value": float(card.get("sell_cost") or 0),
    }


def bridge_state_to_planner_state(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert visible bridge JSON to the state contract used for planner training."""
    hand = raw.get("hand") if isinstance(raw.get("hand"), list) else []
    if len(hand) > MAX_HAND_SLOTS:
        raise ValueError(
            f"Live hand has {len(hand)} cards but planner supports {MAX_HAND_SLOTS}"
        )
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    blind = raw.get("blind") if isinstance(raw.get("blind"), dict) else {}
    hands_remaining = int(raw.get("hands_left") or 0)
    discards_remaining = int(raw.get("discards_left") or 0)
    hand_levels = raw.get("hand_levels") if isinstance(raw.get("hand_levels"), dict) else {}
    poker_hands = {}
    for name in HAND_TYPES:
        values = hand_levels.get(name) if isinstance(hand_levels.get(name), dict) else {}
        poker_hands[name] = {
            "level": values.get("level", 1),
            "chips": values.get("chips", 0),
            "mult": values.get("mult", 0),
            "played": values.get("played", 0),
            "played_this_round": values.get("played_this_round", 0),
        }
    blind_name = str(blind.get("name") or "")
    stake_value = run.get("stake")
    try:
        stake_name = STAKE_NAMES.get(int(stake_value), str(stake_value or "WHITE").upper())
    except (TypeError, ValueError):
        stake_name = str(stake_value or "WHITE").upper()
    live_jokers = raw.get("jokers") if isinstance(raw.get("jokers"), list) else []
    live_consumables = (
        raw.get("consumables") if isinstance(raw.get("consumables"), list) else []
    )
    return {
        "phase": "SELECTING_HAND",
        "round": run.get("round", 0),
        "ante": {"current": run.get("ante", 0), "max": run.get("win_ante", 8)},
        "money": raw.get("money", 0),
        "resources": {
            "hands": {
                "remaining": hands_remaining,
                "max": hands_remaining + int(run.get("hands_played") or 0),
            },
            "discards": {
                "remaining": discards_remaining,
                "max": discards_remaining + int(run.get("discards_used") or 0),
            },
        },
        "blind": {
            "kind": _blind_kind(blind),
            "name": blind_name,
            "effect": BLIND_EFFECTS.get(blind_name, ""),
            "target_score": raw.get("blind_chips", 0),
            "current_score": raw.get("chips", 0),
        },
        "deck": str(run.get("deck") or "RED").upper(),
        "stake": stake_name,
        "poker_hands": poker_hands,
        "hand": {
            "count": len(hand),
            "limit": max(len(hand), int(run.get("hand_size") or 8)),
            "cards": [_live_card(card, index) for index, card in enumerate(hand)],
        },
        "jokers": {
            "count": len(live_jokers),
            "limit": max(len(live_jokers), int(run.get("joker_slots") or 5)),
            "cards": [
                _live_inventory_card(card, index) for index, card in enumerate(live_jokers)
            ],
        },
        "consumables": {
            "count": len(live_consumables),
            "limit": max(len(live_consumables), int(run.get("consumable_slots") or 2)),
            "cards": [
                _live_inventory_card(card, index)
                for index, card in enumerate(live_consumables)
            ],
        },
    }


def load_planner_model(
    model_path: str | Path, device: str = "auto"
) -> tuple[PlannerPolicy, torch.device, dict[str, Any]]:
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    checkpoint = torch.load(Path(model_path), map_location=selected_device, weights_only=True)
    stored_actions = tuple(
        (str(action["kind"]), tuple(action["card_indices"]))
        for action in checkpoint.get("action_vocabulary", [])
    )
    if stored_actions != ACTION_VOCABULARY:
        raise ValueError("Checkpoint action vocabulary does not match this runner")
    model = PlannerPolicy(
        int(checkpoint["observation_dimension"]), int(checkpoint["action_count"])
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, selected_device, checkpoint


def predict_live_action(
    model: PlannerPolicy,
    planner_state: dict[str, Any],
    device: torch.device,
    *,
    stochastic: bool = False,
    temperature: float = 1.0,
    score_plays: bool = True,
) -> dict[str, Any]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    observation = torch.as_tensor(
        encode_planner_state(planner_state), dtype=torch.float32, device=device
    ).unsqueeze(0)
    legal = torch.as_tensor(legal_action_mask(planner_state), device=device).unsqueeze(0)
    if not bool(legal.any().item()):
        raise RuntimeError("Live state has no legal planner actions")
    with torch.no_grad():
        logits = model(observation).masked_fill(~legal, -1e9) / temperature
        probabilities = torch.softmax(logits, dim=1)
        if stochastic:
            policy_action_index = int(torch.multinomial(probabilities[0], 1).item())
        else:
            policy_action_index = int(logits.argmax(dim=1).item())
        top_count = min(5, int(legal.sum().item()))
        top_probabilities, top_indices = probabilities[0].topk(top_count)
    action_index = policy_action_index
    engine_result = None
    policy_kind, _ = ACTION_VOCABULARY[policy_action_index]
    if score_plays and policy_kind == "play":
        scored_candidates = []
        for index, (kind, card_indices) in enumerate(ACTION_VOCABULARY):
            if kind != "play" or not bool(legal[0, index].item()):
                continue
            try:
                result = score_play(planner_state, card_indices)
            except ScoringError:
                continue
            if result.exact:
                scored_candidates.append(
                    (
                        result.expected_score,
                        -len(card_indices),
                        float(probabilities[0, index].item()),
                        -index,
                        index,
                        result,
                    )
                )
        if scored_candidates:
            _, _, _, _, action_index, engine_result = max(scored_candidates)
    kind, card_indices = ACTION_VOCABULARY[action_index]
    top_actions = []
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        top_kind, top_cards = ACTION_VOCABULARY[index]
        top_actions.append(
            {"kind": top_kind, "card_indices": list(top_cards), "probability": probability}
        )
    return {
        "kind": kind,
        "card_indices": list(card_indices),
        "probability": float(probabilities[0, action_index].item()),
        "policy_action": {
            "kind": ACTION_VOCABULARY[policy_action_index][0],
            "card_indices": list(ACTION_VOCABULARY[policy_action_index][1]),
            "probability": float(probabilities[0, policy_action_index].item()),
        },
        "selection_source": "score_rerank" if action_index != policy_action_index else "policy",
        "engine_score": engine_result.to_dict() if engine_result is not None else None,
        "top_actions": top_actions,
    }
