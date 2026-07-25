"""Parse rendered BalatroBench states into planner-ready decision examples."""

from __future__ import annotations

import json
import re
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
IN_BLIND_ACTIONS = {"play", "discard"}
POKER_HAND_NAMES = {
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
}
KNOWN_ENHANCEMENTS = {"BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"}
KNOWN_EDITIONS = {"FOIL", "HOLO", "HOLOGRAPHIC", "POLYCHROME", "NEGATIVE"}
KNOWN_SEALS = {"RED", "BLUE", "GOLD", "PURPLE"}


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


def _write_jsonl(output_file: Any, record: dict[str, Any]) -> None:
    output_file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")


def _number(value: str) -> int | float:
    cleaned = value.replace("$", "").replace(",", "").strip()
    number = float(cleaned)
    return int(number) if number.is_integer() else number


def _field(text: str, label: str) -> str | None:
    match = re.search(rf"(?m)^- \*\*{re.escape(label)}\*\*:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def _parse_count(section: str, noun: str) -> tuple[int | None, int | None]:
    match = re.search(
        rf"The current {re.escape(noun)} count is\s+(\d+)\s*/\s*(\d+)",
        section,
        re.IGNORECASE,
    )
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def _parse_inventory(section: str, noun: str) -> dict[str, Any]:
    count, limit = _parse_count(section, noun)
    cards: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in section.splitlines():
        item_match = re.match(r"^- (\d+):\s+(.+?)\s*$", line)
        if item_match:
            description = item_match.group(2).strip()
            name = description
            effect = ""
            effect_match = re.match(r"^(.*?)\s+\((.*)\)$", description)
            if effect_match:
                name, effect = effect_match.groups()
            current = {
                "index": int(item_match.group(1)),
                "name": name.strip(),
                "effect": effect.strip(),
                "hidden": name.strip().casefold() in {
                    "the joker is face down",
                    "the card is face down",
                },
                "attributes": [],
            }
            cards.append(current)
            continue
        detail_match = re.match(r"^\s+- \*\*(.+?)\*\*(?::\s*(.*))?$", line)
        if current is None or not detail_match:
            continue
        label = detail_match.group(1).strip()
        value = (detail_match.group(2) or "").strip()
        if label.casefold() == "sell value" and value:
            current["sell_value"] = _number(value)
        else:
            current["attributes"].append({"name": label, "value": value})
    return {"count": count, "limit": limit, "cards": cards}


def _parse_poker_hands(section: str) -> dict[str, dict[str, int | float]]:
    hands: dict[str, dict[str, int | float]] = {}
    current_name: str | None = None
    for line in section.splitlines():
        hand_match = re.match(r"^- \*\*(.+?)\*\* \(Level ([\d,.]+)\):", line)
        if hand_match:
            current_name = hand_match.group(1).strip()
            hands[current_name] = {"level": _number(hand_match.group(2))}
            continue
        if current_name is None:
            continue
        value_match = re.match(r"^\s+- \*\*(Chips|Mult)\*\*:\s*([\d,.]+)", line)
        if value_match:
            hands[current_name][value_match.group(1).lower()] = _number(value_match.group(2))
            continue
        played_match = re.match(
            r"^\s+- During this run you have played .+? ([\d,]+) times, "
            r"([\d,]+) of which were played this round\.",
            line,
        )
        if played_match:
            hands[current_name]["played"] = int(played_match.group(1).replace(",", ""))
            hands[current_name]["played_this_round"] = int(
                played_match.group(2).replace(",", "")
            )
    return hands


def _parse_hand(section: str) -> dict[str, Any]:
    count_match = re.search(r"The current card count is\s+(\d+)\s*/\s*(\d+)", section)
    count = int(count_match.group(1)) if count_match else None
    limit = int(count_match.group(2)) if count_match else None
    cards: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in section.splitlines():
        card_match = re.match(r"^- (\d+):\s+(.+?)\s*$", line)
        if card_match:
            index = int(card_match.group(1))
            description = card_match.group(2).strip()
            current = {
                "index": index,
                "hidden": description.casefold() == "the card is face down",
                "stone": description.casefold() == "this is a stone card (no suit and no rank)",
                "key": None,
                "rank": None,
                "suit": None,
                "enhancements": [],
                "editions": [],
                "seals": [],
                "debuffed": False,
                "extra_chips": None,
                "annotations": [],
            }
            identity_match = re.match(
                r"^([A23456789TJQK]) of ([SHDC]) \(`([SHDC]_[A23456789TJQK])`\)$",
                description,
            )
            if identity_match:
                current["rank"], current["suit"], current["key"] = identity_match.groups()
            cards.append(current)
            continue
        detail_match = re.match(r"^\s+- \*\*(.+?)\*\*(?::\s*(.*))?$", line)
        if current is None or not detail_match:
            continue
        label = detail_match.group(1).strip()
        value = (detail_match.group(2) or "").strip()
        upper_label = label.upper()
        if upper_label == "DEBUFF":
            current["debuffed"] = True
        elif upper_label.endswith(" ENHANCEMENT"):
            current["enhancements"].append(label[: -len(" Enhancement")].upper())
        elif upper_label.endswith(" EDITION"):
            current["editions"].append(label[: -len(" Edition")].upper())
        elif upper_label.endswith(" SEAL"):
            current["seals"].append(label[: -len(" Seal")].upper())
        else:
            current["annotations"].append({"name": label, "value": value})
    return {"count": count, "limit": limit, "cards": cards}


def parse_request_state(state_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a BalatroBench Current Game State block without hidden information."""
    errors: list[str] = []
    phase_match = re.search(r"gamestate is ([A-Z0-9_]+)", state_text)
    phase = phase_match.group(1) if phase_match else None

    round_value = _field(state_text, "Round")
    ante_value = _field(state_text, "Ante")
    money_value = _field(state_text, "Money")
    hands_value = _field(state_text, "Hands left")
    discards_value = _field(state_text, "Discards left")
    blind_value = _field(state_text, "Current Blind")
    target_value = _field(state_text, "Target Score")
    score_value = _field(state_text, "Current Score")

    ante_match = re.fullmatch(r"([\d,]+)\s*/\s*([\d,]+)", ante_value or "")
    hands_match = re.fullmatch(r"([\d,]+)\s*/\s*([\d,]+)", hands_value or "")
    discards_match = re.fullmatch(r"([\d,]+)\s*/\s*([\d,]+)", discards_value or "")
    blind_match = re.match(r"^(Small|Big|Boss)(?:\s+\((.*)\))?$", blind_value or "")
    blind_kind = blind_match.group(1).upper() if blind_match else None
    blind_detail = (blind_match.group(2) or "").strip() if blind_match else ""
    blind_name = f"{blind_kind.title()} Blind" if blind_kind in {"SMALL", "BIG"} else None
    blind_effect = ""
    if blind_kind == "BOSS" and blind_detail:
        if ":" in blind_detail:
            blind_name, blind_effect = (part.strip() for part in blind_detail.split(":", 1))
        else:
            blind_name = blind_detail

    hand = _parse_hand(_section(state_text, "Current Hand"))
    poker_hands = _parse_poker_hands(_section(state_text, "Poker Hands"))
    state = {
        "phase": phase,
        "round": _number(round_value) if round_value else None,
        "ante": {
            "current": _number(ante_match.group(1)) if ante_match else None,
            "max": _number(ante_match.group(2)) if ante_match else None,
        },
        "money": _number(money_value) if money_value else None,
        "resources": {
            "hands": {
                "remaining": _number(hands_match.group(1)) if hands_match else None,
                "max": _number(hands_match.group(2)) if hands_match else None,
            },
            "discards": {
                "remaining": _number(discards_match.group(1)) if discards_match else None,
                "max": _number(discards_match.group(2)) if discards_match else None,
            },
        },
        "blind": {
            "kind": blind_kind,
            "name": blind_name,
            "effect": blind_effect,
            "target_score": _number(target_value) if target_value else None,
            "current_score": _number(score_value) if score_value else None,
        },
        "deck": _field(state_text, "Deck"),
        "stake": _field(state_text, "Stake"),
        "jokers": _parse_inventory(_section(state_text, "Jokers"), "Jokers"),
        "consumables": _parse_inventory(
            _section(state_text, "Consumables"), "Consumables"
        ),
        "poker_hands": poker_hands,
        "hand": hand,
    }

    required = {
        "phase": phase,
        "round": state["round"],
        "ante": state["ante"]["current"],
        "money": state["money"],
        "hands_left": state["resources"]["hands"]["remaining"],
        "discards_left": state["resources"]["discards"]["remaining"],
        "blind": blind_kind,
        "target_score": state["blind"]["target_score"],
        "current_score": state["blind"]["current_score"],
        "hand_count": hand["count"],
    }
    errors.extend(f"missing {name}" for name, value in required.items() if value is None)
    if hand["count"] is not None and len(hand["cards"]) != hand["count"]:
        errors.append(
            f"hand count says {hand['count']} but parsed {len(hand['cards'])} cards"
        )
    indices = [card["index"] for card in hand["cards"]]
    if indices != list(range(len(indices))):
        errors.append("hand card indices are not contiguous from zero")
    if len(poker_hands) != 12:
        errors.append(f"expected 12 poker hand definitions but parsed {len(poker_hands)}")
    return state, errors


def _current_blind(state: dict[str, Any]) -> dict[str, Any]:
    blinds = state.get("blinds")
    if not isinstance(blinds, dict):
        return {}
    for blind in blinds.values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return blind
    return {}


def _card_identity(card: Any) -> tuple[str, str | None]:
    if not isinstance(card, dict):
        return "unknown", None
    if card.get("hidden"):
        return "hidden", None
    state = card.get("state")
    if isinstance(state, dict) and (state.get("hidden") or state.get("face_down")):
        return "hidden", None
    if str(card.get("label", "")).casefold() == "stone card":
        return "stone", None
    return "card", str(card.get("key")) if card.get("key") is not None else None


def _modifier_values(card: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {"enhancement": set(), "edition": set(), "seal": set()}
    modifiers = card.get("modifier")
    items = modifiers if isinstance(modifiers, list) else [modifiers]
    for item in items:
        if not isinstance(item, dict):
            continue
        for name in values:
            value = item.get(name)
            if value:
                values[name].add(str(value).upper())
    if str(card.get("label", "")).casefold() == "stone card":
        values["enhancement"].add("STONE")
    return values


def _structured_card_features(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    kind, key = _card_identity(card)
    modifiers = _modifier_values(card)
    state = card.get("state")
    states = state if isinstance(state, list) else [state]
    debuffed = any(isinstance(item, dict) and bool(item.get("debuff")) for item in states)
    return {
        "kind": kind,
        "key": key,
        "enhancements": sorted(modifiers["enhancement"]),
        "editions": sorted(modifiers["edition"]),
        "seals": sorted(modifiers["seal"]),
        "debuffed": debuffed,
    }


def _parsed_card_features(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "hidden" if card["hidden"] else "stone" if card["stone"] else "card",
        "key": card["key"],
        "enhancements": sorted(card["enhancements"] + (["STONE"] if card["stone"] else [])),
        "editions": sorted(card["editions"]),
        "seals": sorted(card["seals"]),
        "debuffed": card["debuffed"],
    }


def _structured_extra_chips(card: Any) -> int | None:
    if not isinstance(card, dict) or _card_identity(card)[0] == "hidden":
        return None
    value = card.get("value")
    effect = str(value.get("effect") or "") if isinstance(value, dict) else ""
    match = re.search(r"\+([\d,]+) extra chips", effect, re.IGNORECASE)
    total_extra = int(match.group(1).replace(",", "")) if match else 0
    modifiers = _modifier_values(card)
    if "BONUS" in modifiers["enhancement"]:
        total_extra -= 30
    return max(0, total_extra)


def enrich_with_structured_card_values(
    parsed: dict[str, Any], structured: dict[str, Any] | None
) -> dict[str, Any]:
    """Add normally visible card values omitted by the rendered request text."""
    enriched = copy.deepcopy(parsed)
    parsed_cards = enriched.get("hand", {}).get("cards", [])
    exact_hand = structured.get("hand") if isinstance(structured, dict) else None
    exact_cards = exact_hand.get("cards") if isinstance(exact_hand, dict) else None
    if not isinstance(exact_cards, list) or len(exact_cards) != len(parsed_cards):
        return enriched
    for parsed_card, exact_card in zip(parsed_cards, exact_cards):
        parsed_card["extra_chips"] = _structured_extra_chips(exact_card)
    return enriched


def _normalized_effect(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def validate_against_structured(
    parsed: dict[str, Any], structured: dict[str, Any]
) -> dict[str, bool]:
    """Compare fields represented in both the rendered and structured states."""
    round_state = structured.get("round") if isinstance(structured.get("round"), dict) else {}
    hand = structured.get("hand") if isinstance(structured.get("hand"), dict) else {}
    exact_cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
    parsed_cards = parsed["hand"]["cards"]
    exact_identities = [_card_identity(card) for card in exact_cards]
    parsed_identities = [
        (
            "hidden" if card["hidden"] else "stone" if card["stone"] else "card",
            card["key"],
        )
        for card in parsed_cards
    ]
    exact_card_features = [_structured_card_features(card) for card in exact_cards]
    parsed_card_features = [_parsed_card_features(card) for card in parsed_cards]
    blind = _current_blind(structured)

    exact_jokers = structured.get("jokers") if isinstance(structured.get("jokers"), dict) else {}
    exact_joker_cards = (
        exact_jokers.get("cards") if isinstance(exact_jokers.get("cards"), list) else []
    )
    exact_consumables = (
        structured.get("consumables")
        if isinstance(structured.get("consumables"), dict)
        else {}
    )
    exact_consumable_cards = (
        exact_consumables.get("cards")
        if isinstance(exact_consumables.get("cards"), list)
        else []
    )
    exact_hands = structured.get("hands") if isinstance(structured.get("hands"), dict) else {}
    comparable_hands = {
        name: {
            key: values.get(key)
            for key in ("level", "chips", "mult", "played", "played_this_round")
        }
        for name, values in exact_hands.items()
        if isinstance(values, dict)
    }

    return {
        "phase": parsed["phase"] == structured.get("state"),
        "round": parsed["round"] == structured.get("round_num"),
        "ante": parsed["ante"]["current"] == structured.get("ante_num"),
        "money": parsed["money"] == structured.get("money"),
        "deck": parsed["deck"] == structured.get("deck"),
        "stake": parsed["stake"] == structured.get("stake"),
        "hands_left": parsed["resources"]["hands"]["remaining"]
        == round_state.get("hands_left"),
        "discards_left": parsed["resources"]["discards"]["remaining"]
        == round_state.get("discards_left"),
        "current_score": parsed["blind"]["current_score"] == round_state.get("chips"),
        "blind_target": parsed["blind"]["target_score"] == blind.get("score"),
        "blind_name": parsed["blind"]["name"] == blind.get("name"),
        "blind_effect": parsed["blind"]["effect"] == str(blind.get("effect") or ""),
        "hand_count": parsed["hand"]["count"] == hand.get("count"),
        "hand_cards": parsed_identities == exact_identities,
        "hand_card_features": parsed_card_features == exact_card_features,
        "joker_count": parsed["jokers"]["count"] == exact_jokers.get("count"),
        "joker_names": [
            None if card["hidden"] else card["name"] for card in parsed["jokers"]["cards"]
        ]
        == [
            None if _card_identity(card)[0] == "hidden" else card.get("label")
            for card in exact_joker_cards
            if isinstance(card, dict)
        ],
        "joker_effects": [
            "" if card["hidden"] else _normalized_effect(card["effect"])
            for card in parsed["jokers"]["cards"]
        ]
        == [
            ""
            if _card_identity(card)[0] == "hidden"
            else _normalized_effect(
                card.get("value", {}).get("effect")
                if isinstance(card.get("value"), dict)
                else ""
            )
            for card in exact_joker_cards
            if isinstance(card, dict)
        ],
        "consumable_count": parsed["consumables"]["count"]
        == exact_consumables.get("count"),
        "consumable_names": [
            None if card["hidden"] else card["name"]
            for card in parsed["consumables"]["cards"]
        ]
        == [
            None if _card_identity(card)[0] == "hidden" else card.get("label")
            for card in exact_consumable_cards
            if isinstance(card, dict)
        ],
        "consumable_effects": [
            "" if card["hidden"] else _normalized_effect(card["effect"])
            for card in parsed["consumables"]["cards"]
        ]
        == [
            ""
            if _card_identity(card)[0] == "hidden"
            else _normalized_effect(
                card.get("value", {}).get("effect")
                if isinstance(card.get("value"), dict)
                else ""
            )
            for card in exact_consumable_cards
            if isinstance(card, dict)
        ],
        "poker_hands": parsed["poker_hands"] == comparable_hands,
    }


def _validate_action(action: dict[str, Any], state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = action.get("name")
    arguments = action.get("arguments")
    cards = arguments.get("cards") if isinstance(arguments, dict) else None
    if name not in IN_BLIND_ACTIONS:
        return ["action is not play or discard"]
    if not isinstance(cards, list) or not 1 <= len(cards) <= 5:
        return ["action must contain between 1 and 5 card indices"]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in cards):
        return ["action card indices must be integers"]
    if len(set(cards)) != len(cards):
        errors.append("action card indices contain duplicates")
    hand_size = len(state["hand"]["cards"])
    if any(index < 0 or index >= hand_size for index in cards):
        errors.append("action card index is outside the parsed hand")
    if name == "discard" and state["resources"]["discards"]["remaining"] == 0:
        errors.append("discard action has no discards remaining")
    return errors


def _run_metadata(record: dict[str, Any]) -> dict[str, Any]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    model = task.get("model") if isinstance(task.get("model"), dict) else {}
    strategy = record.get("strategy") if isinstance(record.get("strategy"), dict) else {}
    return {
        "model": f"{model.get('vendor', 'unknown')}/{model.get('name', 'unknown')}",
        "strategy": strategy.get("name"),
        "seed": task.get("seed"),
        "deck": task.get("deck"),
        "stake": task.get("stake"),
        "run_won": bool(stats.get("run_won")),
        "final_ante": stats.get("final_ante"),
        "final_round": stats.get("final_round"),
    }


def build_planner_dataset(
    *,
    input_dir: str | Path,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join rendered states and actions into planner examples with validation."""
    root = Path(input_dir).resolve()
    output = Path(output_path).resolve() if output_path else root / "planner_examples.jsonl"
    report = Path(manifest_path).resolve() if manifest_path else root / "planner_manifest.json"

    required_files = ["runs.jsonl", "states.jsonl", "request_states.jsonl", "transitions.jsonl"]
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing normalized BalatroBench files: {', '.join(missing)}")

    runs = {record["run_id"]: _run_metadata(record) for record in _iter_jsonl(root / "runs.jsonl")}
    transitions = [
        record for record in _iter_jsonl(root / "transitions.jsonl") if record.get("text_bc_candidate")
    ]
    request_ids = {record.get("request_state_id") for record in transitions}
    pre_state_ids = {record.get("pre_state_id") for record in transitions if record.get("pre_state_id")}

    parsed_requests: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for record in _iter_jsonl(root / "request_states.jsonl"):
        request_state_id = record.get("request_state_id")
        if request_state_id in request_ids:
            parsed_requests[str(request_state_id)] = parse_request_state(str(record.get("state_text", "")))

    structured_states: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(root / "states.jsonl"):
        state_id = record.get("state_id")
        if state_id in pre_state_ids and isinstance(record.get("visible_state"), dict):
            structured_states[str(state_id)] = record["visible_state"]

    counters: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    validation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp_output.open("w", encoding="utf-8", newline="\n") as output_file:
            for transition in transitions:
                counters["examples"] += 1
                request_state_id = str(transition.get("request_state_id"))
                parsed_entry = parsed_requests.get(request_state_id)
                if parsed_entry is None:
                    parsed, parse_errors = {}, ["matching rendered request state is missing"]
                    action_errors = ["cannot validate action without parsed state"]
                else:
                    parsed, parse_errors = parsed_entry
                    action = transition.get("action")
                    action_errors = (
                        _validate_action(action, parsed) if isinstance(action, dict) else ["action is missing"]
                    )

                pre_state_id = transition.get("pre_state_id")
                structured = structured_states.get(str(pre_state_id)) if pre_state_id else None
                checks = validate_against_structured(parsed, structured) if structured and parsed else {}
                if parsed:
                    parsed = enrich_with_structured_card_values(parsed, structured)
                mismatch_fields = sorted(name for name, matched in checks.items() if not matched)
                for field, matched in checks.items():
                    validation_counts[field]["checked"] += 1
                    validation_counts[field]["matched"] += int(matched)

                action = transition.get("action") if isinstance(transition.get("action"), dict) else {}
                arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
                card_indices = arguments.get("cards") if isinstance(arguments.get("cards"), list) else []
                selected_cards = (
                    [parsed["hand"]["cards"][index] for index in card_indices]
                    if parsed and not action_errors
                    else []
                )
                metadata = runs.get(str(transition.get("run_id")), {})
                usable = not parse_errors and not action_errors
                counters["usable_examples"] += int(usable)
                counters["parse_failures"] += int(bool(parse_errors))
                counters["action_failures"] += int(bool(action_errors))
                counters["exact_overlaps"] += int(bool(checks))
                counters["exact_full_matches"] += int(bool(checks) and not mismatch_fields)
                counters["winning_run_examples"] += int(bool(metadata.get("run_won")))
                action_counts[str(action.get("name") or "unknown")] += 1
                model_counts[str(metadata.get("model") or "unknown")] += 1

                example = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": transition.get("transition_id"),
                    "run_id": transition.get("run_id"),
                    "turn": transition.get("turn"),
                    "state": parsed or None,
                    "action": {
                        "kind": action.get("name"),
                        "card_indices": card_indices,
                        "selected_cards": selected_cards,
                    },
                    "outcome": {
                        "score_delta": transition.get("score_delta"),
                        "round_result": transition.get("round_result"),
                        "run_won": metadata.get("run_won"),
                        "final_ante": metadata.get("final_ante"),
                        "final_round": metadata.get("final_round"),
                    },
                    "provenance": {
                        "dataset": "BalatroBench",
                        "model": metadata.get("model"),
                        "strategy": metadata.get("strategy"),
                        "seed": metadata.get("seed"),
                        "source": transition.get("source"),
                        "request_state_id": transition.get("request_state_id"),
                        "structured_pre_state_id": pre_state_id,
                    },
                    "quality": {
                        "usable": usable,
                        "teacher_quality": "unrated_llm_action",
                        "source_tier": "structured_validated" if checks else "rendered_only",
                        "parse_errors": parse_errors,
                        "action_errors": action_errors,
                        "validation_mismatches": mismatch_fields,
                    },
                }
                _write_jsonl(output_file, example)
        temp_output.replace(output)
    except BaseException:
        temp_output.unlink(missing_ok=True)
        raise

    validation = {}
    for field, counts in sorted(validation_counts.items()):
        checked = counts["checked"]
        matched = counts["matched"]
        validation[field] = {
            "checked": checked,
            "matched": matched,
            "match_rate": matched / checked if checked else None,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(root),
        "output": str(output),
        "counts": dict(sorted(counters.items())),
        "actions": dict(sorted(action_counts.items())),
        "models": dict(sorted(model_counts.items())),
        "validation": validation,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="\n") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=True)
        manifest_file.write("\n")
    return manifest


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def audit_planner_dataset(
    *,
    dataset_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit planner examples for privacy, legality, and join consistency."""
    dataset = Path(dataset_path).resolve()
    report = (
        Path(report_path).resolve()
        if report_path
        else dataset.with_name("planner_audit.json")
    )
    counters: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    seeds: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    run_ids: set[str] = set()

    def flag(name: str, example_id: str) -> None:
        issues[name] += 1
        if len(samples[name]) < 5:
            samples[name].append(example_id)

    for example in _iter_jsonl(dataset):
        counters["examples"] += 1
        example_id = str(example.get("example_id") or "")
        if not example_id or example_id in seen_ids:
            flag("missing_or_duplicate_example_id", example_id)
        seen_ids.add(example_id)
        run_ids.add(str(example.get("run_id") or ""))

        state = example.get("state")
        action = example.get("action")
        quality = example.get("quality")
        usable = bool(quality.get("usable")) if isinstance(quality, dict) else False
        counters["usable_examples"] += int(usable)
        counters["rejected_examples"] += int(not usable)
        if not isinstance(state, dict) or not isinstance(action, dict):
            flag("missing_state_or_action", example_id)
            continue
        if _contains_key(state, "seed"):
            flag("seed_inside_policy_state", example_id)
        if state.get("phase") != "SELECTING_HAND":
            flag("wrong_phase", example_id)

        hand = state.get("hand") if isinstance(state.get("hand"), dict) else {}
        hand_cards = hand.get("cards") if isinstance(hand.get("cards"), list) else []
        if hand.get("count") != len(hand_cards):
            flag("hand_count_mismatch", example_id)
        if [card.get("index") for card in hand_cards if isinstance(card, dict)] != list(
            range(len(hand_cards))
        ):
            flag("noncontiguous_hand_indices", example_id)
        for card in hand_cards:
            if not isinstance(card, dict):
                flag("invalid_hand_card", example_id)
                continue
            if card.get("hidden") and any(card.get(name) is not None for name in ("key", "rank", "suit")):
                flag("hidden_card_identity_leak", example_id)
            if not set(card.get("enhancements", [])) <= KNOWN_ENHANCEMENTS:
                flag("unknown_card_enhancement", example_id)
            if not set(card.get("editions", [])) <= KNOWN_EDITIONS:
                flag("unknown_card_edition", example_id)
            if not set(card.get("seals", [])) <= KNOWN_SEALS:
                flag("unknown_card_seal", example_id)

        jokers = state.get("jokers") if isinstance(state.get("jokers"), dict) else {}
        joker_cards = jokers.get("cards") if isinstance(jokers.get("cards"), list) else []
        if jokers.get("count") != len(joker_cards):
            flag("joker_count_mismatch", example_id)
        for joker in joker_cards:
            if isinstance(joker, dict) and joker.get("hidden") and (
                joker.get("effect") or joker.get("name") != "the joker is face down"
            ):
                flag("hidden_joker_identity_leak", example_id)

        consumables = (
            state.get("consumables") if isinstance(state.get("consumables"), dict) else {}
        )
        consumable_cards = (
            consumables.get("cards") if isinstance(consumables.get("cards"), list) else []
        )
        if consumables.get("count") != len(consumable_cards):
            flag("consumable_count_mismatch", example_id)
        if set(state.get("poker_hands", {})) != POKER_HAND_NAMES:
            flag("poker_hand_set_mismatch", example_id)

        kind = action.get("kind")
        indices = action.get("card_indices")
        selected = action.get("selected_cards")
        actions[str(kind)] += 1
        if isinstance(indices, list):
            selected_counts[str(len(indices))] += 1
        action_errors = _validate_action(
            {"name": kind, "arguments": {"cards": indices}}, state
        )
        if action_errors and usable:
            flag("usable_example_has_illegal_action", example_id)
        if not action_errors and not usable:
            flag("legal_example_marked_unusable", example_id)
        expected_selected = (
            [hand_cards[index] for index in indices]
            if isinstance(indices, list)
            and all(isinstance(index, int) and 0 <= index < len(hand_cards) for index in indices)
            else []
        )
        if selected != expected_selected:
            flag("selected_card_join_mismatch", example_id)
        if kind == "play" and state.get("resources", {}).get("hands", {}).get("remaining") == 0:
            flag("play_with_no_hands_remaining", example_id)

        provenance = example.get("provenance")
        if isinstance(provenance, dict) and provenance.get("seed") is not None:
            seeds[str(provenance["seed"])] += 1

    counters["runs"] = len(run_ids)
    counters["unique_seeds"] = len(seeds)
    integrity_errors = sum(issues.values())
    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "status": "passed" if integrity_errors == 0 else "failed",
        "counts": dict(sorted(counters.items())),
        "actions": dict(sorted(actions.items())),
        "selected_card_counts": dict(sorted(selected_counts.items())),
        "seeds": dict(sorted(seeds.items())),
        "integrity_errors": integrity_errors,
        "issues": dict(sorted(issues.items())),
        "issue_samples": dict(sorted(samples.items())),
        "notes": [
            "A passed audit verifies structure and visibility, not that LLM actions are optimal.",
            "Split training data by seed and run, never by individual decision row.",
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(audit, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")
    return audit
