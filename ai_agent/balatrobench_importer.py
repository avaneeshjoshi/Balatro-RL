"""Stream BalatroBench runs into normalized run, state, and transition tables."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
IN_BLIND_ACTIONS = {"play", "discard"}
ACTION_PHASES = {
    "play": {"SELECTING_HAND"},
    "discard": {"SELECTING_HAND"},
    "pack": {"SMODS_BOOSTER_OPENED"},
    "select_blind": {"BLIND_SELECT"},
    "skip": {"BLIND_SELECT"},
    "cash_out": {"ROUND_EVAL"},
    "buy": {"SHOP"},
    "reroll": {"SHOP"},
    "next_round": {"SHOP"},
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    try:
        input_file = path.open("r", encoding="utf-8")
    except OSError as exc:
        yield 0, None, str(exc)
        return
    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, str(exc)
                continue
            if not isinstance(value, dict):
                yield line_number, None, "record is not a JSON object"
                continue
            yield line_number, value, None


def _jsonl_record_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return sum(1 for line in input_file if line.strip())
    except OSError:
        return 0


def _is_hidden(card: dict[str, Any]) -> bool:
    state = card.get("state")
    if isinstance(state, dict):
        return bool(state.get("hidden") or state.get("face_down"))
    if isinstance(state, list):
        for item in state:
            if isinstance(item, str) and item.lower() in {"hidden", "face_down", "facedown"}:
                return True
            if isinstance(item, dict) and (item.get("hidden") or item.get("face_down")):
                return True
    return False


def _sanitize_hand_card(card: Any) -> Any:
    if not isinstance(card, dict):
        return card
    if not _is_hidden(card):
        return copy.deepcopy(card)
    return {
        "hidden": True,
        "state": copy.deepcopy(card.get("state")),
    }


def _card_sort_key(card: Any) -> str:
    if not isinstance(card, dict):
        return json.dumps(card, sort_keys=True, separators=(",", ":"))
    sortable = copy.deepcopy(card)
    sortable.pop("state", None)
    return json.dumps(sortable, sort_keys=True, separators=(",", ":"))


def sanitize_visible_state(raw: dict[str, Any]) -> dict[str, Any]:
    """Remove metadata/leaks while retaining normally visible state information."""
    state = copy.deepcopy(raw)
    state.pop("seed", None)
    state.pop("won", None)

    hand = state.get("hand")
    if isinstance(hand, dict) and isinstance(hand.get("cards"), list):
        hand["cards"] = [_sanitize_hand_card(card) for card in hand["cards"]]

    jokers = state.get("jokers")
    if isinstance(jokers, dict) and isinstance(jokers.get("cards"), list):
        # Amber Acorn and similar effects can conceal joker identities. Keep the
        # slot and hidden marker, but never retain information the player cannot see.
        jokers["cards"] = [_sanitize_hand_card(card) for card in jokers["cards"]]

    deck = state.get("cards")
    if isinstance(deck, dict) and isinstance(deck.get("cards"), list):
        # BalatroBench emits the remaining deck in draw order. Deck composition is
        # visible to a player, but that order is not, so erase it deterministically.
        deck["cards"] = sorted(deck["cards"], key=_card_sort_key)
    return state


def _extract_message(response_record: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response_record["response"]["body"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}


def _parse_arguments(value: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, dict):
        return copy.deepcopy(value), None
    if not isinstance(value, str):
        return {}, "tool arguments are not an object or JSON string"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return {}, f"invalid tool argument JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments do not decode to an object"
    return parsed, None


def extract_tool_calls(
    response_record: dict[str, Any],
    *,
    include_reasoning: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    message = _extract_message(response_record)
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return [], ["response has no tool calls"]
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, raw_call in enumerate(raw_calls):
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if not isinstance(function, dict):
            errors.append(f"tool call {index} has no function object")
            continue
        arguments, error = _parse_arguments(function.get("arguments"))
        if error:
            errors.append(f"tool call {index}: {error}")
        reasoning = arguments.pop("reasoning", None)
        original_name = str(function.get("name") or "")
        normalized_name = original_name.removeprefix("functions.")
        call = {
            "name": normalized_name,
            "arguments": arguments,
        }
        if normalized_name != original_name:
            call["original_name"] = original_name
        if include_reasoning and reasoning is not None:
            call["reasoning"] = reasoning
        calls.append(call)
    return calls, errors


def extract_request_state(request_record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract only the rendered current-state block, excluding seed and tool docs."""
    body = request_record.get("body")
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return None, "request has no messages list"
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else part
            if not isinstance(text, str) or "# Current Game State" not in text:
                continue
            state_text = text[text.index("# Current Game State") :]
            state_text = state_text.split("# Tools available", maxsplit=1)[0]
            state_text = re.sub(
                r"(?m)^- \*\*Seed\*\*:.*(?:\r?\n)?",
                "",
                state_text,
            ).strip()
            return state_text, None
    return None, "request has no Current Game State block"


def _request_phase(state_text: str | None) -> str | None:
    if not state_text:
        return None
    match = re.search(r"gamestate is ([A-Z0-9_]+)", state_text)
    return match.group(1) if match else None


def _text_action_is_well_formed(action: dict[str, Any] | None) -> bool:
    if not action or action.get("name") not in IN_BLIND_ACTIONS:
        return False
    arguments = action.get("arguments")
    cards = arguments.get("cards") if isinstance(arguments, dict) else None
    return bool(
        isinstance(cards, list)
        and 1 <= len(cards) <= 5
        and all(isinstance(card, int) and not isinstance(card, bool) and card >= 0 for card in cards)
        and len(set(cards)) == len(cards)
    )


def _phase(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    value = state.get("state")
    return str(value) if value is not None else None


def _pre_state_is_structured(action_name: str, previous_state: dict[str, Any] | None) -> bool:
    phase = _phase(previous_state)
    expected = ACTION_PHASES.get(action_name)
    if expected is not None:
        return phase in expected
    # For less common actions, a repeated stable phase is useful canonical data,
    # but it is not automatically eligible for the in-blind BC subset.
    return previous_state is not None and phase is not None


def _hand_size(state: dict[str, Any]) -> int:
    hand = state.get("hand")
    if not isinstance(hand, dict):
        return 0
    cards = hand.get("cards")
    if isinstance(cards, list):
        return len(cards)
    try:
        return int(hand.get("count", 0))
    except (TypeError, ValueError):
        return 0


def validate_in_blind_action(
    action: dict[str, Any] | None,
    pre_state: dict[str, Any] | None,
    post_state: dict[str, Any] | None,
) -> list[str]:
    if not action or action.get("name") not in IN_BLIND_ACTIONS:
        return ["not an in-blind play/discard action"]
    errors: list[str] = []
    if pre_state is None:
        errors.append("no structured pre-action state")
        return errors
    if _phase(pre_state) != "SELECTING_HAND":
        errors.append("pre-action phase is not SELECTING_HAND")
    if post_state is None:
        errors.append("no structured post-action state")
    arguments = action.get("arguments")
    cards = arguments.get("cards") if isinstance(arguments, dict) else None
    if not isinstance(cards, list):
        errors.append("cards argument is not a list")
        return errors
    if not 1 <= len(cards) <= 5:
        errors.append("action must select between 1 and 5 cards")
    if any(isinstance(card, bool) or not isinstance(card, int) for card in cards):
        errors.append("card indices must be integers")
        return errors
    if len(set(cards)) != len(cards):
        errors.append("card indices contain duplicates")
    hand_size = _hand_size(pre_state)
    if any(card < 0 or card >= hand_size for card in cards):
        errors.append("card index is outside the pre-action hand")
    if action.get("name") == "discard":
        round_state = pre_state.get("round")
        discards_left = round_state.get("discards_left", 0) if isinstance(round_state, dict) else 0
        try:
            if int(discards_left) <= 0:
                errors.append("discard action has no discards remaining")
        except (TypeError, ValueError):
            errors.append("discards_left is not numeric")
    return errors


def _score_delta(pre_state: dict[str, Any] | None, post_state: dict[str, Any] | None) -> float | None:
    if not isinstance(pre_state, dict) or not isinstance(post_state, dict):
        return None
    if (pre_state.get("round_num"), pre_state.get("ante_num")) != (
        post_state.get("round_num"),
        post_state.get("ante_num"),
    ):
        return None
    pre_round = pre_state.get("round")
    post_round = post_state.get("round")
    if not isinstance(pre_round, dict) or not isinstance(post_round, dict):
        return None
    try:
        return float(post_round.get("chips", 0)) - float(pre_round.get("chips", 0))
    except (TypeError, ValueError):
        return None


def _round_result(post_state: dict[str, Any] | None) -> str | None:
    phase = _phase(post_state)
    if phase == "GAME_OVER":
        return "lost"
    if phase == "ROUND_EVAL":
        return "won"
    return None


def _write_jsonl(output_file: Any, record: dict[str, Any]) -> None:
    output_file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")


def _run_id(root: Path, run_dir: Path) -> str:
    return run_dir.relative_to(root).as_posix()


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("task.json"))


def import_balatrobench(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    include_reasoning: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Import all discovered runs and return the generated manifest."""
    source = Path(source_root).resolve()
    destination = Path(output_dir).resolve()
    runs = discover_runs(source)
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        runs = runs[:max_runs]
    if not runs:
        raise FileNotFoundError(f"No BalatroBench task.json files found under {source}")

    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "runs": destination / "runs.jsonl",
        "states": destination / "states.jsonl",
        "request_states": destination / "request_states.jsonl",
        "transitions": destination / "transitions.jsonl",
    }
    temporary_paths = {name: path.with_suffix(path.suffix + ".tmp") for name, path in output_paths.items()}

    counters: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()

    try:
        with ExitStack() as stack:
            outputs = {
                name: stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
                for name, path in temporary_paths.items()
            }
            for run_index, run_dir in enumerate(runs, start=1):
                identifier = _run_id(source, run_dir)
                task = _load_json(run_dir / "task.json")
                stats = _load_json(run_dir / "stats.json")
                strategy = _load_json(run_dir / "strategy.json")
                model = task.get("model") if isinstance(task.get("model"), dict) else {}
                model_name = f"{model.get('vendor', 'unknown')}/{model.get('name', 'unknown')}"
                model_counts[model_name] += 1
                counters["runs"] += 1
                counters["run_wins"] += int(bool(stats.get("run_won")))
                run_record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": identifier,
                    "source": {
                        "dataset": "BalatroBench",
                        "root": str(source),
                        "relative_dir": identifier,
                    },
                    "task": task,
                    "stats": stats,
                    "strategy": strategy,
                }
                _write_jsonl(outputs["runs"], run_record)

                states_path = run_dir / "gamestates.jsonl"
                responses_path = run_dir / "responses.jsonl"
                requests_path = run_dir / "requests.jsonl"
                state_record_count = _jsonl_record_count(states_path)
                response_record_count = _jsonl_record_count(responses_path)
                alignment_exact = state_record_count == response_record_count
                counters["aligned_runs"] += int(alignment_exact)
                counters["unaligned_runs"] += int(not alignment_exact)
                previous_raw_state: dict[str, Any] | None = None
                previous_state_id: str | None = None
                request_lookup: dict[str, dict[str, Any]] = {}
                for request_line, request_record, request_error in _iter_jsonl(requests_path):
                    request_record = request_record or {}
                    request_id = request_record.get("custom_id")
                    request_state_text, request_state_error = extract_request_state(request_record)
                    request_state_id = (
                        f"{identifier}#request-state-{request_line:05d}"
                        if request_state_text is not None
                        else None
                    )
                    lookup_key = str(request_id) if request_id else f"@line-{request_line}"
                    request_lookup[lookup_key] = {
                        "line": request_line,
                        "request_id": request_id,
                        "request_state_id": request_state_id,
                        "state_text": request_state_text,
                        "error": request_error or request_state_error,
                    }
                    if request_state_id is not None:
                        _write_jsonl(
                            outputs["request_states"],
                            {
                                "schema_version": SCHEMA_VERSION,
                                "request_state_id": request_state_id,
                                "run_id": identifier,
                                "turn": request_line,
                                "request_id": request_id,
                                "source": {
                                    "file": requests_path.relative_to(source).as_posix(),
                                    "line": request_line,
                                },
                                "phase": _request_phase(request_state_text),
                                "state_text": request_state_text,
                            },
                        )
                        counters["request_states"] += 1
                    else:
                        counters["missing_request_states"] += 1
                state_rows = _iter_jsonl(states_path)
                response_rows = _iter_jsonl(responses_path)
                for turn, row_group in enumerate(zip_longest(state_rows, response_rows), start=1):
                    state_item, response_item = row_group
                    state_line, raw_state, state_error = state_item or (0, None, None)
                    response_line, response_record, response_error = response_item or (
                        0,
                        None,
                        None,
                    )
                    state_id = (
                        f"{identifier}#state-{state_line:05d}"
                        if raw_state is not None
                        else None
                    )
                    if raw_state is not None and state_id is not None:
                        visible_state = sanitize_visible_state(raw_state)
                        phase = _phase(raw_state) or "unknown"
                        phase_counts[phase] += 1
                        _write_jsonl(
                            outputs["states"],
                            {
                                "schema_version": SCHEMA_VERSION,
                                "state_id": state_id,
                                "run_id": identifier,
                                "turn": turn,
                                "source": {
                                    "file": states_path.relative_to(source).as_posix(),
                                    "line": state_line,
                                },
                                "visible_state": visible_state,
                            },
                        )
                        counters["states"] += 1
                    elif state_error:
                        counters["state_errors"] += 1
                    elif state_item is None:
                        counters["unmatched_response_rows"] += 1

                    response_record = response_record or {}
                    response_request_id = response_record.get("custom_id")
                    request_key = (
                        str(response_request_id)
                        if response_request_id
                        else f"@line-{response_line}"
                    )
                    request_entry = request_lookup.pop(request_key, None)
                    request_line = int(request_entry["line"]) if request_entry else 0
                    request_state_id = request_entry.get("request_state_id") if request_entry else None
                    request_state_text = request_entry.get("state_text") if request_entry else None
                    request_error = request_entry.get("error") if request_entry else "response has no matching request"
                    tool_calls, extraction_errors = extract_tool_calls(
                        response_record,
                        include_reasoning=include_reasoning,
                    )
                    primary_action = tool_calls[0] if len(tool_calls) == 1 else None
                    action_name = str(primary_action.get("name") or "") if primary_action else ""
                    action_counts[action_name or "<none_or_multiple>"] += 1
                    structured_pre = alignment_exact and _pre_state_is_structured(
                        action_name,
                        previous_raw_state,
                    )
                    pre_state = previous_raw_state if structured_pre else None
                    pre_id = previous_state_id if structured_pre else None
                    aligned_post_state = raw_state if alignment_exact else None
                    aligned_post_id = state_id if alignment_exact else None
                    validation_errors = validate_in_blind_action(
                        primary_action,
                        pre_state,
                        aligned_post_state,
                    )
                    is_in_blind = action_name in IN_BLIND_ACTIONS
                    bc_candidate = is_in_blind and not validation_errors
                    text_bc_candidate = (
                        _request_phase(request_state_text) == "SELECTING_HAND"
                        and _text_action_is_well_formed(primary_action)
                    )
                    counters["transitions"] += 1
                    counters["in_blind_actions"] += int(is_in_blind)
                    counters["bc_candidates"] += int(bc_candidate)
                    counters["text_bc_candidates"] += int(text_bc_candidate)
                    counters["missing_structured_pre_state"] += int(pre_id is None)
                    counters["response_errors"] += int(bool(response_error or extraction_errors))

                    transition = {
                        "schema_version": SCHEMA_VERSION,
                        "transition_id": f"{identifier}#transition-{response_line:05d}",
                        "run_id": identifier,
                        "turn": response_line,
                        "request_id": response_request_id,
                        "source": {
                            "response_file": responses_path.relative_to(source).as_posix(),
                            "response_line": response_line,
                            "request_file": requests_path.relative_to(source).as_posix(),
                            "request_line": request_line,
                        },
                        "pre_state_id": pre_id,
                        "post_state_id": aligned_post_id,
                        "request_state_id": request_state_id,
                        "pre_state_source": "previous_gamestate" if pre_id else "request_state_text",
                        "alignment": "exact" if alignment_exact else "unresolved_failed_calls",
                        "tool_calls": tool_calls,
                        "action": primary_action,
                        "phase_before": _phase(pre_state),
                        "phase_after": _phase(aligned_post_state),
                        "score_delta": _score_delta(pre_state, aligned_post_state),
                        "round_result": _round_result(aligned_post_state),
                        "bc_candidate": bc_candidate,
                        "text_bc_candidate": text_bc_candidate,
                        "validation_errors": validation_errors,
                        "import_errors": [
                            error
                            for error in [
                                state_error,
                                response_error,
                                request_error,
                                *extraction_errors,
                            ]
                            if error
                        ],
                    }
                    _write_jsonl(outputs["transitions"], transition)
                    previous_raw_state = raw_state if alignment_exact else None
                    previous_state_id = state_id if alignment_exact else None

                for request_entry in sorted(request_lookup.values(), key=lambda item: int(item["line"])):
                    request_line = int(request_entry["line"])
                    request_state_text = request_entry.get("state_text")
                    request_state_id = request_entry.get("request_state_id")
                    _write_jsonl(
                        outputs["transitions"],
                        {
                            "schema_version": SCHEMA_VERSION,
                            "transition_id": f"{identifier}#request-only-{request_line:05d}",
                            "run_id": identifier,
                            "turn": request_line,
                            "request_id": request_entry.get("request_id"),
                            "source": {
                                "response_file": responses_path.relative_to(source).as_posix(),
                                "response_line": 0,
                                "request_file": requests_path.relative_to(source).as_posix(),
                                "request_line": request_line,
                            },
                            "pre_state_id": None,
                            "post_state_id": None,
                            "request_state_id": request_state_id,
                            "pre_state_source": "request_state_text",
                            "alignment": "request_only",
                            "tool_calls": [],
                            "action": None,
                            "phase_before": None,
                            "phase_after": None,
                            "score_delta": None,
                            "round_result": None,
                            "bc_candidate": False,
                            "text_bc_candidate": False,
                            "validation_errors": ["response is missing"],
                            "import_errors": [
                                error
                                for error in [request_entry.get("error"), "request has no response"]
                                if error
                            ],
                        },
                    )
                    counters["transitions"] += 1
                    counters["missing_structured_pre_state"] += 1
                    counters["response_errors"] += 1
                    action_counts["<request_only>"] += 1

                if run_index % 25 == 0 or run_index == len(runs):
                    print(
                        f"Imported {run_index}/{len(runs)} runs | "
                        f"states={counters['states']} transitions={counters['transitions']} "
                        f"bc_candidates={counters['bc_candidates']}"
                    )
        for name, temporary_path in temporary_paths.items():
            temporary_path.replace(output_paths[name])
    except BaseException:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source),
        "output_dir": str(destination),
        "include_reasoning": include_reasoning,
        "counts": dict(sorted(counters.items())),
        "actions": dict(sorted(action_counts.items())),
        "phases": dict(sorted(phase_counts.items())),
        "models": dict(sorted(model_counts.items())),
        "files": {name: path.name for name, path in output_paths.items()},
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=True)
    return manifest
