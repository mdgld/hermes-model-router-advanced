"""
model-router plugin
===================
Automatic cost-aware model routing.

Runtime source of truth:
  - active profile `model_router.yaml`

Built-in defaults only apply when that file does not exist yet or is invalid.

Features:
  - Pre-LLM classification on every turn
  - Explicit tier request detection: T3, tier4, t2, etc.
  - /model and /t1-/t5 set a session pin until /auto or a fresh session
  - Mid-loop self-escalation on repeated tool errors
  - Post-heavy-work de-escalation after T4/T5 completes
  - Status bar: shows [Tx] prefix dynamically in front of model name
  - Ambiguous multi-tier mention guard: "T1 T2 T3" = discussing tiers, not requesting one
"""

from __future__ import annotations

import copy
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model definitions used only for bootstrap/fallback
# ---------------------------------------------------------------------------

DEFAULT_ROUTER_CONFIG = {
    "classifier": {
        "provider": "openrouter",
        "model": "qwen/qwen3.5-flash-02-23",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "timeout": 30,
        "extra_body": {"enable_caching": True},
    },
    "tiers": {
        1: {
            "label": "T1 Flash",
            "emoji": "⚡",
            "model": "qwen/qwen3.5-flash-02-23",
            "reasoning": None,
            "role": "fast triage and cheap helper",
            "best_for": [
                "Short acknowledgements",
                "Intent classification",
                "Status checks",
                "Title generation",
            ],
        },
        2: {
            "label": "T2 DeepSeek",
            "emoji": "🔹",
            "model": "deepseek/deepseek-v4-flash",
            "reasoning": None,
            "role": "default daily-driver",
            "best_for": [
                "Default day-to-day work",
                "Documentation and drafting",
                "Standard coding and research",
            ],
        },
        3: {
            "label": "T3 MiniMax",
            "emoji": "🔷",
            "model": "minimax/minimax-m2.7",
            "reasoning": None,
            "role": "strong reasoning and synthesis",
            "best_for": [
                "Debugging",
                "Code review",
                "Large-document synthesis",
                "Complex analysis",
            ],
        },
        4: {
            "label": "T4 Haiku",
            "emoji": "🔸",
            "model": "anthropic/claude-haiku-4-5",
            "reasoning": None,
            "role": "deliberate fast planner",
            "best_for": [
                "Architecture",
                "Migration planning",
                "Complex multi-step design",
                "Nuanced code review",
            ],
        },
        5: {
            "label": "T5 Sonnet",
            "emoji": "🔶",
            "model": "anthropic/claude-sonnet-4-6",
            "reasoning": "medium",
            "role": "expensive deep-think mode",
            "best_for": [
                "Security-sensitive analysis",
                "Algorithmic optimization",
                "High-stakes reasoning",
            ],
        },
    },
}

_router_config: dict[str, Any] = copy.deepcopy(DEFAULT_ROUTER_CONFIG)
TIERS: dict[int, dict[str, Any]] = {}
MODEL_TO_TIER: dict[str, int] = {}
FLASH_MODEL = ""
FLASH_PROVIDER = ""
_TIER_LABELS: dict[int, tuple[str, str]] = {}


def _get_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _router_config_path() -> Path:
    return _get_hermes_home() / "model_router.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_router_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    merged = _deep_merge(DEFAULT_ROUTER_CONFIG, raw or {})
    normalized_tiers: dict[int, dict[str, Any]] = {}
    raw_tiers = merged.get("tiers", {})
    for tier_num in range(1, 6):
        tier_defaults = copy.deepcopy(DEFAULT_ROUTER_CONFIG["tiers"][tier_num])
        override = raw_tiers.get(tier_num, raw_tiers.get(str(tier_num), {}))
        if not isinstance(override, dict):
            override = {}
        tier_defaults.update(override)
        normalized_tiers[tier_num] = tier_defaults
    merged["tiers"] = normalized_tiers
    return merged


def _apply_router_config(config: dict[str, Any]) -> None:
    global _router_config, TIERS, MODEL_TO_TIER, FLASH_MODEL, FLASH_PROVIDER, _TIER_LABELS

    _router_config = config
    TIERS = {
        tier_num: {
            "model": meta["model"],
            "reasoning": meta.get("reasoning"),
            "label": meta.get("label", f"T{tier_num}"),
            "emoji": meta.get("emoji", ""),
            "role": meta.get("role", ""),
            "best_for": meta.get("best_for", []),
        }
        for tier_num, meta in config["tiers"].items()
    }

    model_to_tier: dict[str, int] = {}
    for tier_num in sorted(TIERS):
        model = TIERS[tier_num]["model"]
        model_to_tier.setdefault(model, tier_num)
    MODEL_TO_TIER = model_to_tier

    classifier = config.get("classifier", {})
    FLASH_MODEL = classifier.get("model", DEFAULT_ROUTER_CONFIG["classifier"]["model"])
    FLASH_PROVIDER = classifier.get("provider", DEFAULT_ROUTER_CONFIG["classifier"]["provider"])

    _TIER_LABELS = {
        tier_num: (meta.get("emoji", ""), meta.get("label", f"T{tier_num}"))
        for tier_num, meta in config["tiers"].items()
    }


def _load_router_config() -> None:
    path = _router_config_path()
    if not path.exists():
        _apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))
        return

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("model_router.yaml must be a mapping")
        _apply_router_config(_normalize_router_config(raw))
        logger.info("model-router: loaded config from %s", path)
    except Exception as exc:
        logger.warning("model-router: failed to load %s: %s -- using defaults", path, exc)
        _apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))


_apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))

# ---------------------------------------------------------------------------
# Explicit tier/model request detection
# ---------------------------------------------------------------------------

# Detect standalone tier mentions: "T3", "t2", "T4", "(T5)", etc.
# Avoids false positives inside words: "cat5", "step3", "T100"
_TIER_STANDALONE_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=\())[tT]([1-5])(?:\b|(?=\)))"
)
_TIER_WORD_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=\())tier\s*([1-5])", re.IGNORECASE
)


def _detect_explicit_tier(msg: str) -> int | None:
    """Return an explicitly requested tier, or None if ambiguous/absent.

    Supported syntax:
      T3, t2, T4, T5      — standalone Tx notation (case-insensitive)
      tier4, tier 3       — word form (case-insensitive, TIER 3, Tier5, etc.)

    Model name keywords (sonnet, haiku, flash, plus, qwen...) are intentionally
    NOT supported here — use /model for explicit model selection. Those keywords
    cause more false positives than true routing signals.

    Ambiguity guard: if the message mentions 3+ distinct tier numbers
    (e.g. "T1 T2 T3 T4 T5" or discussing the routing table) we treat it
    as an informational mention, not a routing directive, and defer to Flash.
    This prevents the common false positive where the user asks *about* tiers.

    Two mentions: takes the HIGHEST (e.g. "T3 vs T4 approach" → T4).
    """
    # Collect all Tx / tierN mentions
    all_tier_mentions = set(
        int(m) for m in _TIER_STANDALONE_RE.findall(msg)
    )
    word_mentions = set(
        int(m) for m in _TIER_WORD_RE.findall(msg)
    )
    all_tier_mentions |= word_mentions

    # Ambiguity guard: 3+ distinct tier numbers = discussing tiers, not requesting
    if len(all_tier_mentions) >= 3:
        logger.debug(
            "model-router: %d distinct tier mentions in msg -- ambiguous, deferring to Flash",
            len(all_tier_mentions),
        )
        return None

    # Exactly one or two mentions -> take the HIGHEST one as the intent
    # (e.g. "compare T3 vs T4 approach" -> T4 makes sense as a cap)
    if all_tier_mentions:
        tier_num = max(all_tier_mentions)
        logger.info("model-router: explicit tier request detected: T%d", tier_num)
        return tier_num

    return None


# ---------------------------------------------------------------------------
# Fast-path heuristic: obvious short ACKs -> T1, no Flash call
# ---------------------------------------------------------------------------

_ACK_RE = re.compile(
    r"^(ok|okay|thanks|thank you|thx|got it|understood|sure|yes|no|yep|nope|"
    r"alright|cool|great|nice|perfect|done|noted|ack|"
    r"dzięki|tak|nie|rozumiem|gotowe|spoko|super|świetnie|"
    r"hello|hi|hey|cześć|hej)"
    r"[!?.]*$",
    re.IGNORECASE,
)

def _is_obvious_ack(msg: str) -> bool:
    words = msg.strip().split()
    if len(words) > 6:
        return False
    return bool(_ACK_RE.match(msg.strip()))


# ---------------------------------------------------------------------------
# Flash classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are a model routing classifier. Your only job is to assign a tier number (1-5) to the user's message based on task complexity.

Tier definitions:
1 = Ultra-cheap: short acknowledgements, status checks, title generation, <4 word messages
2 = Default: normal day-to-day work, content drafting, Q&A, file operations, SEO research, standard coding
3 = Stronger: debugging with multiple causes, code review, large-doc summarization, nuanced analysis
4 = Planner: architecture decisions, system design, multi-step plans, migration planning, workflow design
5 = Deep-think: security analysis, cryptography, algorithmic optimization (complexity/performance), financial modelling, high-stakes with many interacting constraints

Rules:
- When unsure between two tiers, pick the LOWER one
- Tier 5 only for truly security-critical or algorithmically dense tasks
- Polish and English messages treated equally
- Consider the INTENT not just keywords

Respond with ONLY a single digit: 1, 2, 3, 4, or 5. Nothing else."""


def _classify_with_flash(user_message: str, conversation_history: list) -> int:
    """Call Flash to classify turn complexity. Returns tier 1-5."""
    try:
        from agent.auxiliary_client import call_llm

        # Include last 2 assistant turns as context (cheap, small)
        context_turns = []
        assistant_count = 0
        for msg in reversed(conversation_history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    context_turns.insert(0, {"role": "assistant", "content": content[:300]})
                    assistant_count += 1
                    if assistant_count >= 2:
                        break

        messages = [{"role": "system", "content": _CLASSIFIER_SYSTEM}]
        if context_turns:
            messages.append({
                "role": "user",
                "content": "[Recent conversation context]\n"
                           + "\n".join(f"Assistant: {m['content']}" for m in context_turns),
            })
            messages.append({"role": "assistant", "content": "Understood."})

        messages.append({"role": "user", "content": user_message[:800]})

        response = call_llm(
            task="triage_specifier",
            messages=messages,
            max_tokens=3,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        digit = re.search(r"[1-5]", raw)
        if digit:
            return int(digit.group())
        return 2
    except Exception as exc:
        logger.warning("model-router: Flash classification failed: %s -- defaulting T2", exc)
        return 2


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

_session_last:    dict[str, tuple[str, int]] = {}  # session_id -> (msg, tier)
_session_manual:  dict[str, str] = {}              # session_id -> model set manually by user
_session_pinned:  dict[str, bool] = {}             # session_id -> True when user used /model or /tN
                                                    # plugin stays OFF until /auto or /new
_last_tier:       dict[str, int] = {}              # session_id -> current assigned tier
_base_tier:       dict[str, int] = {}              # session_id -> tier for this turn (before escalation)
_tool_errors:     dict[str, int] = {}              # session_id -> consecutive tool error count this turn
_escalated:       dict[str, bool] = {}             # session_id -> True if we already escalated mid-turn
_state_lock = threading.Lock()

def get_last_tier(session_id: str) -> int:
    with _state_lock:
        return _last_tier.get(session_id, 0)


def is_session_pinned(session_id: str) -> bool:
    """True if auto-routing is disabled for this session (user used /model or /tN)."""
    with _state_lock:
        return _session_pinned.get(session_id, False)


def _get_cached_tier(session_id: str, msg: str) -> int | None:
    with _state_lock:
        entry = _session_last.get(session_id)
        if entry and entry[0] == msg:
            return entry[1]
    return None


def _set_cached_tier(session_id: str, msg: str, tier: int) -> None:
    with _state_lock:
        # Cache one entry per session (msg, tier) for dedup.
        # Keeps only the CURRENT message — new messages will replace old ones.
        # This avoids unbounded growth while preserving dedup for immediate reruns.
        _session_last[session_id] = (msg, tier)


def _is_manual_override(session_id: str, current_model: str) -> bool:
    """True if user manually pinned a model via /model or /tN command.

    Pinned sessions are FULLY blocked — plugin does not touch model or
    reasoning_config until the user explicitly calls /auto (or /new).
    """
    with _state_lock:
        return _session_pinned.get(session_id, False)


def _record_router_set(session_id: str) -> None:
    """Called after the plugin itself applies a tier switch.
    Does NOT clear the pin — only /auto can clear a pin.
    Clears the _session_manual sentinel (used only for auto-detection)."""
    with _state_lock:
        _session_manual.pop(session_id, None)


def notify_manual_override(session_id: str, model: str) -> None:
    """Called when we auto-detect a /model change mid-session.
    Sets the session pin so we stop routing for this session."""
    with _state_lock:
        _session_manual[session_id] = model
        _session_pinned[session_id] = True


def pin_session(session_id: str, model: str) -> None:
    """Explicit pin — called from /model, /t1-/t5 slash commands.
    Immediately halts auto-routing for this session."""
    with _state_lock:
        _session_manual[session_id] = model
        _session_pinned[session_id] = True
    logger.info("model-router: session %s pinned to %s", session_id, model)


def unpin_session(session_id: str) -> None:
    """Called from /auto slash command. Re-enables auto-routing."""
    with _state_lock:
        _session_manual.pop(session_id, None)
        _session_pinned.pop(session_id, None)
        # Also clear cache so next turn re-classifies fresh
        _session_last.pop(session_id, None)
    logger.info("model-router: session %s unpinned, auto-routing resumed", session_id)


# ---------------------------------------------------------------------------
# Status bar injection
# ---------------------------------------------------------------------------

def _patch_status_bar(cli) -> None:
    """Monkey-patch cli._get_status_bar_snapshot to inject [Tx] tier prefix.

    Called once after we have a cli reference. Safe to call multiple times --
    guards against double-patching via _router_patched sentinel.
    """
    if getattr(cli, "_router_patched", False):
        return
    original_snapshot = cli._get_status_bar_snapshot.__func__  # unbound

    def _patched_snapshot(self_cli):
        snap = original_snapshot(self_cli)
        # Read current tier from our registry
        session_id = getattr(getattr(self_cli, "agent", None), "session_id", None) or ""
        tier = get_last_tier(session_id)
        if tier:
            prefix = f"[T{tier}] "
            # Read the model name from agent.model directly so the badge
            # and the model name always match (agent.model may have changed
            # since the original snapshot was captured).
            agent = getattr(self_cli, "agent", None)
            current_model = getattr(agent, "model", "") if agent else ""
            if not current_model:
                # Fallback to tier's default model if agent.model is somehow empty
                current_model = TIERS.get(tier, {}).get("model", "")
            # Extract just the short name (e.g. "claude-sonnet-4-6" from "anthropic/claude-sonnet-4-6")
            name_only = current_model.split("/")[-1] if current_model else "unknown"
            snap["model_short"] = prefix + name_only
        return snap

    import types
    cli._get_status_bar_snapshot = types.MethodType(_patched_snapshot, cli)
    cli._router_patched = True
    logger.debug("model-router: status bar patch applied")


# ---------------------------------------------------------------------------
# Apply model switch helper
# ---------------------------------------------------------------------------

def _apply_tier(session_id: str, target_tier: int, current_model: str, source: str = "") -> None:
    """Switch agent.model + reasoning_config to target_tier. Emits badge."""
    global _manager_ref

    if _manager_ref is None:
        return

    target_model     = TIERS[target_tier]["model"]
    target_reasoning = TIERS[target_tier]["reasoning"]
    current_tier     = MODEL_TO_TIER.get(current_model, 2)

    # Apply status bar patch lazily (needs cli ref)
    try:
        cli = _manager_ref._cli_ref
        if cli:
            _patch_status_bar(cli)
    except Exception:
        pass

    try:
        cli   = _manager_ref._cli_ref
        agent = getattr(cli, "agent", None) if cli else None
        if agent is None:
            return

        old_model   = agent.model
        agent.model = target_model

        if target_reasoning:
            agent.reasoning_config = {"effort": target_reasoning}
        else:
            agent.reasoning_config = None

        _record_router_set(session_id)

        with _state_lock:
            _last_tier[session_id] = target_tier

        emoji, label = _TIER_LABELS.get(target_tier, ("", f"T{target_tier}"))
        src_tag = f" [{source}]" if source else ""
        tier_msg = f"{emoji} model-router: {label} ({target_model.split('/')[-1]}){src_tag}"

        _vprint = getattr(cli, "_vprint", None) or getattr(cli, "_cprint", None)
        if _vprint:
            try:
                _vprint(tier_msg)
            except Exception:
                logger.info(tier_msg)
        else:
            logger.info(tier_msg)

        logger.info(
            "model-router: T%d->T%d | %s -> %s%s",
            current_tier, target_tier,
            old_model, target_model,
            f" [{source}]" if source else "",
        )
    except Exception as exc:
        logger.warning("model-router: failed to apply switch: %s", exc)


# ---------------------------------------------------------------------------
# Hook: pre_llm_call  (fires before EVERY LLM call in the loop)
# ---------------------------------------------------------------------------

def on_pre_llm_call(
    *,
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = True,
    model: str = "",
    session_id: str = "",
    platform: str = "",
    **kwargs: Any,
) -> None:

    if _manager_ref is None:
        return

    msg = user_message.strip()
    if not msg:
        return

    history = conversation_history or []

    # Get live agent.model instead of relying on stale 'model' parameter
    # (model parameter is from plugin hook context and may lag behind actual agent state)
    actual_model = model
    try:
        cli = _manager_ref._cli_ref
        if cli:
            agent = getattr(cli, "agent", None)
            if agent:
                actual_model = agent.model
    except Exception as exc:
        logger.debug("model-router: failed to get live agent.model: %s", exc)

    # Respektuj ręczny override
    if _is_manual_override(session_id, actual_model):
        logger.debug("model-router: manual override active (%s), skipping", actual_model)
        return

    # ── Determine whether this is a "new user turn" vs tool-loop iteration ───
    #
    # IMPORTANT: Hermes passes is_first_turn=True ONLY when conversation_history
    # is empty (i.e. the very first turn of the entire session).  Every subsequent
    # user turn arrives with is_first_turn=False because conversation_history is
    # already populated.  We therefore cannot rely on is_first_turn to detect a
    # new user message.
    #
    # Instead we track the last message text we classified per session.  If msg
    # differs from the last classified message this is definitely a new user turn
    # and we re-classify.  If msg is the same we are inside the tool-calling loop
    # for the same turn and we only need to confirm escalation state.
    with _state_lock:
        last_entry = _session_last.get(session_id)
        is_new_user_turn = (last_entry is None) or (last_entry[0] != msg)

    if is_new_user_turn:
        # Reset per-turn error counters
        with _state_lock:
            _tool_errors[session_id] = 0
            _escalated[session_id]   = False

        # 1. Explicit user request? (e.g. "T3 pls", "a T2 jesteś tu?")
        explicit = _detect_explicit_tier(msg)
        if explicit is not None:
            target_tier = explicit
            _set_cached_tier(session_id, msg, target_tier)
            logger.info("model-router: explicit T%d request honoured", target_tier)
        elif _is_obvious_ack(msg):
            target_tier = 1
            _set_cached_tier(session_id, msg, target_tier)
        else:
            # No explicit hint — classify with Flash
            target_tier = _classify_with_flash(msg, history)
            _set_cached_tier(session_id, msg, target_tier)

        with _state_lock:
            _base_tier[session_id] = target_tier
            _last_tier[session_id] = target_tier

        # Apply only if different from current model
        logger.debug(
            "model-router: turn T%d -> model=%s vs actual=%s",
            target_tier, TIERS[target_tier]["model"], actual_model
        )
        if TIERS[target_tier]["model"] != actual_model:
            logger.info(
                "model-router: switching T%d (was T%d / %s)",
                target_tier, MODEL_TO_TIER.get(actual_model, 0), actual_model.split("/")[-1]
            )
            _apply_tier(session_id, target_tier, actual_model, source="")
        else:
            # Same model -- still update status bar patch and tier registry
            with _state_lock:
                _last_tier[session_id] = target_tier
            try:
                cli = _manager_ref._cli_ref
                if cli:
                    _patch_status_bar(cli)
            except Exception:
                pass

    else:
        # ── Tool-loop iteration for the same user turn ────────────────────────
        # Self-escalation is handled by on_post_tool_call.  Here we just confirm
        # that any mid-loop escalation is still reflected in the active model.
        with _state_lock:
            tier = _last_tier.get(session_id, 2)
        if TIERS[tier]["model"] != actual_model:
            # Escalation was applied mid-loop — confirm the switch is still live
            _apply_tier(session_id, tier, actual_model, source="escalated")


# ---------------------------------------------------------------------------
# Hook: post_tool_call  (fires after every tool execution inside the loop)
# ---------------------------------------------------------------------------

# How many consecutive tool errors trigger a self-escalation
_ESCALATION_ERROR_THRESHOLD = 2

def on_post_tool_call(
    *,
    tool_name: str = "",
    result: str | None = None,
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Self-escalation: if the agent keeps hitting tool errors, bump one tier.

    Only escalates ONCE per turn (guards against ping-pong).
    Only escalates if currently below T4 (T4/T5 are already the strongest).
    Resets error counter on success.
    Does NOT escalate if session is pinned by user via /model or /tN.
    """
    if _manager_ref is None or not session_id:
        return

    # Respect session pin — never auto-escalate a pinned session
    with _state_lock:
        if _session_pinned.get(session_id, False):
            return

    # Detect error in result
    is_error = False
    if result is not None:
        result_lower = result[:500].lower()
        if (
            '"error"' in result_lower
            or '"failed"' in result_lower
            or result.startswith("Error")
            or "exit_code" in result_lower and '"exit_code": ' in result_lower
            and not '"exit_code": 0' in result_lower
            and not '"exit_code": null' in result_lower
        ):
            is_error = True

    with _state_lock:
        if is_error:
            _tool_errors[session_id] = _tool_errors.get(session_id, 0) + 1
        else:
            _tool_errors[session_id] = 0  # reset on success

        error_count = _tool_errors.get(session_id, 0)
        already_escalated = _escalated.get(session_id, False)
        current_tier = _last_tier.get(session_id, 2)

    if (
        is_error
        and error_count >= _ESCALATION_ERROR_THRESHOLD
        and current_tier < 4
    ):
        new_tier = min(current_tier + 1, 4)
        with _state_lock:
            _escalated[session_id] = True
            _last_tier[session_id] = new_tier
            _tool_errors[session_id] = 0  # Reset counter so next 2 errors can trigger another escalation

        try:
            cli   = _manager_ref._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            current_model = agent.model if agent else TIERS[current_tier]["model"]
        except Exception:
            current_model = TIERS[current_tier]["model"]

        logger.info(
            "model-router: self-escalating T%d->T%d after %d tool errors",
            current_tier, new_tier, error_count,
        )
        _apply_tier(session_id, new_tier, current_model, source="auto-escalate")


# ---------------------------------------------------------------------------
# Hook: post_llm_call  (fires after every LLM response in the loop)
# ---------------------------------------------------------------------------

def on_post_llm_call(
    *,
    model: str = "",
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Two responsibilities:

    1. Detect user manual /model change and pin the session (stop auto-routing).
    2. De-escalate: if we escalated mid-turn, restore base tier now that
       the heavy work is done (fires after the final response is complete).
       Only de-escalates if is_first_turn would be True next call, i.e.
       we detect this is the FINAL response (no pending tool calls).
       We approximate this by checking if the current call was NOT mid-loop.
    """
    global _manager_ref
    if _manager_ref is None:
        return

    try:
        cli   = _manager_ref._cli_ref
        agent = getattr(cli, "agent", None) if cli else None
        if agent is None:
            return

        # 1. Detect external manual model change (e.g. user ran /model mid-session)
        #    agent.model was changed outside of our _apply_tier call.
        #    We detect this by comparing agent.model to what we last set.
        with _state_lock:
            already_pinned = _session_pinned.get(session_id, False)
            router_last_manual = _session_manual.get(session_id)

        if not already_pinned:
            # Check if agent.model is now something we didn't set
            with _state_lock:
                last_router_tier = _last_tier.get(session_id, 0)
            expected_model = TIERS[last_router_tier]["model"] if last_router_tier else None

            if expected_model and agent.model != expected_model and agent.model != model:
                # Model changed between our last set and now — user did /model
                notify_manual_override(session_id, agent.model)
                logger.info(
                    "model-router: detected manual model change to %s -- pinning session, auto-routing paused",
                    agent.model,
                )
                return

        # If already pinned, do nothing (no de-escalation either)
        if already_pinned:
            return

        # 2. De-escalate after heavy work completes
        with _state_lock:
            was_escalated = _escalated.get(session_id, False)
            base = _base_tier.get(session_id, 2)
            current = _last_tier.get(session_id, 2)

        if was_escalated and current > base:
            logger.info(
                "model-router: de-escalating T%d->T%d after heavy work completed",
                current, base,
            )
            with _state_lock:
                _escalated[session_id] = False
                _last_tier[session_id] = base
            _apply_tier(session_id, base, agent.model, source="de-escalate")

    except Exception:
        pass


def _apply_reasoning(effort: str | None) -> None:
    """No-op helper stub."""
    pass


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _manager_ref
    _load_router_config()
    _manager_ref = ctx._manager
    ctx.register_hook("pre_llm_call",   on_pre_llm_call)
    ctx.register_hook("post_llm_call",  on_post_llm_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)

    # Expose public API on the PluginManager so slash commands (/t1-/t5, /auto)
    # can call us via get_plugin_manager().router_apply_tier(...).
    # NOTE: ctx is a PluginContext facade, but cli.py reads from the manager
    # directly via get_plugin_manager(). Setting on ctx._manager ensures the
    # attributes are visible where the handlers look for them.
    mgr = ctx._manager
    mgr.router_pin_session   = pin_session
    mgr.router_unpin_session = unpin_session
    mgr.router_apply_tier    = _apply_tier_by_num
    mgr.router_is_pinned     = is_session_pinned
    mgr.router_get_tier      = get_last_tier
    mgr.router_get_tier_meta = get_tier_meta

    logger.info(
        "model-router: registered -- Flash T1-T5 routing | explicit hints | "
        "self-escalation | de-escalation | status bar [Tx] | /tN + /auto commands"
    )


def get_tier_meta(tier_num: int) -> dict[str, Any]:
    """Return model metadata for a tier so the CLI does not hardcode slugs."""
    if tier_num not in TIERS:
        return {}
    return dict(_router_config["tiers"][tier_num])


def _apply_tier_by_num(session_id: str, tier_num: int, current_model: str) -> None:
    """Public entry point for /t1-/t5 slash commands.

    Sets the tier, applies the model switch, and pins the session so
    auto-routing does not override the choice.
    """
    if tier_num not in TIERS:
        logger.warning("model-router: invalid tier %d requested", tier_num)
        return
    target_model = TIERS[tier_num]["model"]
    pin_session(session_id, target_model)
    with _state_lock:
        _last_tier[session_id] = tier_num
        _base_tier[session_id] = tier_num
    _apply_tier(session_id, tier_num, current_model, source="user-pin")
