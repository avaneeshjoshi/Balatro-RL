"""
Gymnasium environment for Balatro. Reads state from bridge state.json,
encodes to observation, and sends actions via command.json.

Scope: blind-round card play only (small/big/boss blind). No power-up,
shop, or joker selection — the agent only chooses which cards to play
or discard to maximize chip score within each round.

Provider safeguards (avoid "silent killers"):
  1. Race condition: step() does NOT return until a *new* state is seen.
     We poll for state.json with seq > last_seq (and retry once after timeout),
     so the game has time to process the command and write an updated state.
  2. Stale state: Bridge puts a monotonic "seq" in every state.json. We only
     accept state with seq > last_seq before returning from step(), so the
     observation is always from after our last action.
  3. Invalid actions: We clamp card indices to 1..hand_size, cap "play"
     at 5 cards, and convert discard to play when no discards remain.
     The Lua bridge independently validates the same game-side constraints.
"""
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym  
import numpy as np 

from .feature_encoder import (
    encode_state,
    load_state_json,
    HAND_TYPES_ORDER,
    OBS_DIM_V1,
    OBS_DIM_V2,
    observation_dimension,
)

# Action: [play_or_discard, card0, card1, ..., card7]; play=0, discard=1; card_i = 0/1
NUM_CARD_SLOTS = 8
ACTION_DIM = 1 + NUM_CARD_SLOTS  # 9
# Obs: 6 + 8*3 (phase..hand) + len(HAND_TYPES_ORDER)*3 (hand levels)
OBS_DIM = OBS_DIM_V1  # Backward-compatible alias for the original model.
MAX_PLAY_CARDS = 5  # Balatro rule: at most 5 cards per play

# Reward shaping: hand-type bonuses when bridge sends last_hand_type (encourages complex hands)
HAND_BONUS: dict[str, float] = {
    "High Card": 0.0,
    "Pair": 0.005,
    "Two Pair": 0.01,
    "Three of a Kind": 0.02,
    "Straight": 0.03,
    "Flush": 0.04,
    "Full House": 0.06,
    "Four of a Kind": 0.08,
    "Straight Flush": 0.12,
    "Royal Flush": 0.15,
}


class BalatroEnv(gym.Env):
    """
    Balatro hand-selection environment (blind rounds only, no power-ups).
    Agent observes normalized state and chooses which cards to play or
    discard (binary mask over up to 8 cards) to maximize chip score.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bridge_dir: str | Path,
        state_timeout: float = 15.0,
        advance_timeout: float = 120.0,
        step_delay: float = 0.1,
        reward_scale: float = 1e-4,
        reward_discard_penalty: float = 0.001,
        reward_win_bonus: float = 1.0,
        reward_loss_penalty: float = 1.0,
        reward_hand_bonus: bool = True,
        auto_advance: bool = False,
        observation_version: int = 1,
        num_card_slots: int = NUM_CARD_SLOTS,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.bridge_dir = Path(bridge_dir)
        self.state_path = self.bridge_dir / "state.json"
        self.command_path = self.bridge_dir / "command.json"
        self.state_timeout = state_timeout
        if advance_timeout <= 0:
            raise ValueError("advance_timeout must be positive")
        self.advance_timeout = advance_timeout
        self.step_delay = step_delay
        self.reward_scale = reward_scale
        self.reward_discard_penalty = reward_discard_penalty
        self.reward_win_bonus = reward_win_bonus
        self.reward_loss_penalty = reward_loss_penalty
        self.reward_hand_bonus = reward_hand_bonus
        self.auto_advance = auto_advance
        self.observation_version = observation_version
        self.obs_dim = observation_dimension(observation_version)
        if num_card_slots < 1:
            raise ValueError("num_card_slots must be positive")
        self.num_card_slots = int(num_card_slots)

        self._last_chips: float = 0.0
        self._last_raw: dict[str, Any] | None = None
        self._last_seq: int = -1

        # Observation: normalized vector (see feature_encoder)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        # Action: 0=play, 1=discard; then one bit per configured hand slot.
        self.action_space = gym.spaces.MultiDiscrete(
            [2] * (1 + self.num_card_slots)
        )

    def _encode(self, raw: dict[str, Any]) -> np.ndarray:
        return encode_state(raw, self.observation_version)

    def _read_state(self) -> dict[str, Any] | None:
        return load_state_json(self.state_path)

    def _wait_for_state(self) -> dict[str, Any] | None:
        """Wait for state.json to exist and be valid. No requirement that it's 'new'."""
        deadline = time.monotonic() + self.state_timeout
        while time.monotonic() < deadline:
            raw = self._read_state()
            if raw is not None:
                return raw
            time.sleep(self.step_delay)
        return None

    def _is_playable_state(self, raw: dict[str, Any]) -> bool:
        """True if state has a non-empty hand (we're in SELECTING_HAND with cards)."""
        hand = raw.get("hand") or []
        return len(hand) > 0

    def _round_result(self, raw: dict[str, Any] | None) -> str | None:
        """Return won/lost for terminal bridge states, otherwise None."""
        if raw is None:
            return None
        result = str(raw.get("round_result") or "").strip().lower()
        if result in {"won", "lost"}:
            return result
        chips = float(raw.get("chips", 0))
        blind_chips = float(raw.get("blind_chips", 0))
        if blind_chips > 0 and chips >= blind_chips:
            return "won"
        return None

    def _wait_for_new_state(self, after_seq: int) -> tuple[dict[str, Any] | None, str]:
        """
        Wait for state.json with seq > after_seq and either a playable hand or
        a terminal round result. Non-terminal transition states have hand=[] and
        are skipped until Balatro settles.
        """
        _seq_warned = getattr(self, "_seq_missing_warned", False)
        deadline = time.monotonic() + self.state_timeout
        current_seq = after_seq
        while time.monotonic() < deadline:
            raw = self._read_state()
            if raw is not None:
                if "seq" not in raw and not _seq_warned:
                    import warnings
                    warnings.warn(
                        "state.json has no 'seq' key — Python will never see 'new' state and will timeout. "
                        "Restart Balatro so it loads the updated bridge (main.lua with state_seq).",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._seq_missing_warned = True
                seq = raw.get("seq", 0)
                if seq > current_seq:
                    current_seq = seq
                    if self._is_playable_state(raw) or self._round_result(raw):
                        return raw, "new_state"
            time.sleep(self.step_delay)
        retry_deadline = time.monotonic() + 5.0
        while time.monotonic() < retry_deadline:
            raw = self._read_state()
            if raw is not None:
                seq = raw.get("seq", 0)
                if seq > current_seq:
                    current_seq = seq
                    if self._is_playable_state(raw) or self._round_result(raw):
                        return raw, "new_state"
            time.sleep(self.step_delay)
        return None, "timeout"

    def _write_command(self, action: str, card_indices: list[int]) -> None:
        payload = {"action": action, "cards": card_indices}
        with open(self.command_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    def _action_to_command(self, action: np.ndarray) -> tuple[str, list[int]]:
        """
        Convert MultiDiscrete action to (play|discard, 1-based card indices).
        Clamps to current hand size, caps play at MAX_PLAY_CARDS (5), and
        converts discard to play when no discards remain.
        """
        act = np.asarray(action).flatten()
        hand_size = self.num_card_slots
        if self._last_raw is not None:
            hand = self._last_raw.get("hand") or []
            hand_size = max(1, len(hand))
        play_or_discard = int(act[0]) if act.size > 0 else 0
        action_type = "play" if play_or_discard == 0 else "discard"
        if (
            action_type == "discard"
            and self._last_raw is not None
            and int(self._last_raw.get("discards_left", 0)) <= 0
        ):
            action_type = "play"
        indices = []
        for i in range(self.num_card_slots):
            if i + 1 < act.size and int(act[i + 1]) == 1:
                idx = i + 1  # 1-based for Lua
                if idx <= hand_size:
                    indices.append(idx)
        if not indices:
            indices = [1]
        # Balatro allows at most 5 cards per play
        if action_type == "play" and len(indices) > MAX_PLAY_CARDS:
            indices = indices[:MAX_PLAY_CARDS]
        return action_type, indices

    def _compute_reward(
        self,
        raw: dict[str, Any],
        action_type: str,
    ) -> float:
        """
        Shaped reward: chip delta + optional hand bonus + discard penalty + win bonus.
        Helps credit assignment (agent learns hand types and beating the blind).
        """
        chips = float(raw.get("chips", 0))
        hands_left = int(raw.get("hands_left", 0))
        previous_hands = (
            int(self._last_raw.get("hands_left", 0))
            if self._last_raw is not None
            else hands_left
        )
        played_hand = (
            action_type == "play"
            and (chips > self._last_chips or hands_left < previous_hands)
        )
        r = (chips - self._last_chips) * self.reward_scale
        if action_type == "discard":
            r -= self.reward_discard_penalty
        if self.reward_hand_bonus and played_hand:
            hand_name = str(raw.get("last_hand_played") or raw.get("last_hand_type") or "").strip()
            if hand_name:
                r += HAND_BONUS.get(hand_name, 0.0)
        round_result = self._round_result(raw)
        if round_result == "won":
            r += self.reward_win_bonus
        elif round_result == "lost":
            r -= self.reward_loss_penalty
        return r

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        raw = self._wait_for_state()
        if raw is None:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            self._last_raw = None
            self._last_chips = 0.0
            self._last_seq = -1
            return obs, {"raw_state": None, "message": "no state file", "truncated_reason": "timeout"}
        if not self._is_playable_state(raw):
            if self.auto_advance:
                self._write_command("advance", [])
            deadline = time.monotonic() + (
                self.advance_timeout if self.auto_advance else self.state_timeout
            )
            while time.monotonic() < deadline:
                raw = self._read_state()
                if raw is not None and self._is_playable_state(raw):
                    break
                time.sleep(self.step_delay)
        if raw is None or not self._is_playable_state(raw):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            self._last_raw = None
            self._last_chips = 0.0
            self._last_seq = -1
            return obs, {"raw_state": None, "message": "no playable state (no hand)", "truncated_reason": "timeout"}
        self._last_raw = raw
        self._last_chips = float(raw.get("chips", 0))
        self._last_seq = int(raw.get("seq", 0))
        obs = self._encode(raw)
        return obs, {"raw_state": raw}

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, int):
            action = np.array([action], dtype=np.int64)
        action = np.atleast_1d(action)
        requested_action_type = (
            "discard" if action.size > 0 and int(action[0]) == 1 else "play"
        )
        action_type, card_indices = self._action_to_command(action)
        self._write_command(action_type, card_indices)
        time.sleep(self.step_delay * 2)
        raw, stop_reason = self._wait_for_new_state(self._last_seq)
        if raw is None:
            obs = self._encode(self._last_raw) if self._last_raw else np.zeros(self.obs_dim, dtype=np.float32)
            return obs, 0.0, False, True, {
                "raw_state": None,
                "truncated_reason": stop_reason,
                "terminated_reason": None,
            }
        obs = self._encode(raw)
        chips = float(raw.get("chips", 0))
        reward = self._compute_reward(raw, action_type)
        round_result = self._round_result(raw)
        terminated = round_result is not None
        self._last_chips = chips
        self._last_raw = raw
        self._last_seq = int(raw.get("seq", 0))
        return obs, reward, terminated, False, {
            "raw_state": raw,
            "truncated_reason": None,
            "terminated_reason": f"blind_{round_result}" if terminated else None,
            "action_type": action_type,
            "card_indices": card_indices,
            "action_overridden": requested_action_type != action_type,
        }
