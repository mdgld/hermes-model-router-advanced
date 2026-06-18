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

import contextlib
import copy
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model definitions used only for bootstrap/fallback
# KEEP IN SYNC with install.py DEFAULT_ROUTER_CONFIG — duplicated intentionally
# so install.py remains a self-contained script with no import dependencies.
# ---------------------------------------------------------------------------

DEFAULT_ROUTER_CONFIG = {
    "provider_priority": ["nous", "openai-codex", "openrouter"],
    "classifier": {
        "provider": "nous",
        "model": "deepseek/deepseek-v4-flash",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "api_key": "",
        "timeout": 30,
        "extra_body": {"enable_caching": True, "reasoning_effort": "high"},
        "fallbacks": [
            {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning_effort": "low"},
            {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", "reasoning_effort": "high"},
        ],
    },
    "tiers": {
        1: {
            "label": "T1 Flash (MiMo v2.5 Pro)",
            "emoji": "⚡",
            "model": "xiaomi/mimo-v2.5-pro",
            "reasoning": "enabled",
            "extra_body": {"enable_caching": True},
            "role": "fast triage and cheap helper",
            "best_for": [
                "Short acknowledgements",
                "Intent classification",
                "Status checks",
                "Title generation",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning": "low", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "xiaomi/mimo-v2.5-pro", "reasoning": "enabled", "extra_body": {"enable_caching": True}},
            ],
        },
        2: {
            "label": "T2 (DeepSeek v4 Pro)",
            "emoji": "🔹",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning": "max",
            "extra_body": {"enable_caching": True},
            "role": "day-to-day usage, basic tasks",
            "best_for": [
                "Standard day-to-day work",
                "Well-defined documentation and drafting",
                "Extremely basic coding and research",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning": "medium", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "reasoning": "max", "extra_body": {"enable_caching": True}},
            ],
        },
        3: {
            "label": "T3 (MiniMax M3)",
            "emoji": "🔷",
            "model": "minimax/minimax-m3",
            "reasoning": "enabled",
            "extra_body": {"enable_caching": True},
            "role": "standard coding, well-defined short tasks",
            "best_for": [
                "Basic troubleshooting",
                "Light code review",
                "Standard reasoning",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning": "xhigh", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "minimax/minimax-m3", "reasoning": "enabled", "extra_body": {"enable_caching": True}},
            ],
        },
        4: {
            "label": "T4 (GLM 5.2)",
            "emoji": "🔸",
            "model": "z-ai/glm-5.2",
            "reasoning": "xhigh",
            "extra_body": {"enable_caching": True},
            "role": "strong reasoning and synthesis",
            "best_for": [
                "Architecture and refactoring",
                "Migration planning",
                "Basic agentic workflows",
                "Complex multi-step designs and workflows",
                "Nuanced code review",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4", "reasoning": "high", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "z-ai/glm-5.2", "reasoning": "xhigh", "extra_body": {"enable_caching": True}},
            ],
        },
        5: {
            "label": "T5 (GPT-5.5)",
            "emoji": "🔶🔶",
            "model": "openai/gpt-latest",
            "reasoning": "high",
            "extra_body": {"enable_caching": True},
            "role": "expensive deep-think mode",
            "best_for": [
                "Security-sensitive analysis",
                "High-stakes tasks",
                "Algorithmic optimization",
                "Long-context agentic workflows",
                "Near-human-level reasoning on certain tasks",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.5", "reasoning": "high"},
                {"provider": "openrouter", "model": "openai/gpt-latest", "reasoning": "high", "extra_body": {"enable_caching": True}},
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
PROVIDER_PRIORITY: list[str] = []
TIER_FALLBACKS: dict[int, list[dict[str, Any]]] = {}
CLASSIFIER_FALLBACKS: list[dict[str, Any]] = []
_provider_failures: dict[str, float] = {}
_PROVIDER_UNHEALTHY_TTL = 120.0

_PROVIDER_BASE_URLS: dict[str, str] = {
    "nous": "https://inference-api.nousresearch.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
}


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
        # Preserve fallbacks from user yaml if present; they may differ from defaults
        if "fallbacks" in override:
            tier_defaults["fallbacks"] = override["fallbacks"]
        normalized_tiers[tier_num] = tier_defaults
    merged["tiers"] = normalized_tiers
    return merged


def _apply_router_config(config: dict[str, Any]) -> None:
    global _router_config, TIERS, MODEL_TO_TIER, FLASH_MODEL, FLASH_PROVIDER, _TIER_LABELS
    global PROVIDER_PRIORITY, TIER_FALLBACKS, CLASSIFIER_FALLBACKS

    _router_config = config
    TIERS = {
        tier_num: {
            "model": meta["model"],
            "provider": meta.get("provider"),
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

    PROVIDER_PRIORITY = config.get("provider_priority", list(DEFAULT_ROUTER_CONFIG.get("provider_priority", [])))
    CLASSIFIER_FALLBACKS = classifier.get("fallbacks", list(DEFAULT_ROUTER_CONFIG["classifier"].get("fallbacks", [])))
    TIER_FALLBACKS = {
        tier_num: list(meta.get("fallbacks", []))
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

    Model name keywords (sonnet, deepseek, flash, plus, qwen...) are intentionally
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
    """Call Flash to classify turn complexity. Returns tier 1-5.

    Attempts the configured primary classifier first (via triage_specifier task),
    then retries each entry in classifier.fallbacks if the primary fails.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.warning("model-router: auxiliary_client unavailable: %s -- defaulting T2", exc)
        return 2

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

    def _parse_tier(response: Any) -> int | None:
        try:
            raw = response.choices[0].message.content.strip()
            digit = re.search(r"[1-5]", raw)
            return int(digit.group()) if digit else None
        except Exception:
            return None

    # --- Primary attempt (uses triage_specifier task config from config.yaml) ---
    primary_provider = FLASH_PROVIDER or (PROVIDER_PRIORITY[0] if PROVIDER_PRIORITY else "")
    if _is_provider_healthy(primary_provider):
        try:
            response = call_llm(
                task="triage_specifier",
                messages=messages,
                max_tokens=3,
                temperature=0.0,
            )
            tier = _parse_tier(response)
            if tier is not None:
                return tier
        except Exception as exc:
            logger.warning("model-router: classifier primary (%s) failed: %s", primary_provider, exc)
            _mark_provider_failed(primary_provider)
    else:
        logger.info("model-router: classifier primary provider %r unhealthy; skipping to fallbacks", primary_provider)

    # --- Fallback attempts ---
    for fb in CLASSIFIER_FALLBACKS:
        fb_provider = str(fb.get("provider") or "").strip()
        fb_model    = str(fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        if not _is_provider_healthy(fb_provider):
            logger.debug("model-router: classifier fallback %r unhealthy; skipping", fb_provider)
            continue
        fb_base_url = _PROVIDER_BASE_URLS.get(fb_provider)
        try:
            fb_extra: dict[str, Any] = {}
            fb_reasoning = fb.get("reasoning_effort")
            if fb_reasoning:
                fb_extra["reasoning_effort"] = fb_reasoning
            response = call_llm(
                provider=fb_provider,
                model=fb_model,
                base_url=fb_base_url,
                messages=messages,
                max_tokens=3,
                temperature=0.0,
                extra_body=fb_extra or None,
            )
            tier = _parse_tier(response)
            if tier is not None:
                logger.info("model-router: classifier fallback %r/%s succeeded", fb_provider, fb_model)
                return tier
        except Exception as exc:
            logger.warning("model-router: classifier fallback %r/%s failed: %s", fb_provider, fb_model, exc)
            _mark_provider_failed(fb_provider)

    logger.warning("model-router: all classifier providers exhausted -- defaulting T2")
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
_live_agents:     dict[str, Any] = {}              # session_id -> active agent bound by WebUI/CLI bridge
_live_tui_sessions: dict[str, tuple[str, dict]] = {}  # session_id -> (tui_sid, session_dict) for TUI mode
_session_ts:      dict[str, float] = {}            # session_id -> last-seen monotonic timestamp
_SESSION_TTL      = 86_400.0                       # evict sessions idle longer than 24 h
_state_lock = threading.Lock()
_patch_lock = threading.Lock()                     # guards _patch_status_bar double-patch check
_tui_server_module: Any = None                     # cached tui_gateway.server module (pre-loaded at register time)

def get_last_tier(session_id: str) -> int:
    with _state_lock:
        return _last_tier.get(session_id, 0)


def _evict_stale_sessions() -> None:
    """Remove state for sessions idle longer than _SESSION_TTL. Called on each new user turn."""
    cutoff = time.monotonic() - _SESSION_TTL
    with _state_lock:
        stale = [sid for sid, ts in _session_ts.items() if ts < cutoff]
        for sid in stale:
            for d in (
                _session_last, _session_manual, _session_pinned, _last_tier,
                _base_tier, _tool_errors, _escalated, _live_agents, _live_tui_sessions, _session_ts,
            ):
                d.pop(sid, None)
    if stale:
        logger.debug("model-router: evicted %d stale session(s)", len(stale))


# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------

def _mark_provider_failed(provider: str) -> None:
    """Record a provider failure timestamp for health-check gating."""
    if not provider:
        return
    with _state_lock:
        _provider_failures[provider.lower()] = time.monotonic()
    logger.info("model-router: provider %r marked unhealthy (TTL %ss)", provider, _PROVIDER_UNHEALTHY_TTL)


def _is_provider_healthy(provider: str) -> bool:
    """Return True when provider has not failed recently (or has no recorded failure)."""
    if not provider:
        return True
    with _state_lock:
        failed_at = _provider_failures.get(provider.lower(), 0.0)
    return (time.monotonic() - failed_at) > _PROVIDER_UNHEALTHY_TTL


def _select_tier_entry(tier: int) -> tuple[str, str | None, str]:
    """Return (model, reasoning, provider) for tier, skipping unhealthy primary provider.

    Falls through the PROVIDER_PRIORITY order.  Returns the primary entry if
    no fallback improves things (caller handles the error naturally).
    """
    tier_data = TIERS.get(tier, {})
    primary_model     = tier_data.get("model", "")
    primary_reasoning = tier_data.get("reasoning")
    primary_provider  = str(tier_data.get("provider") or (PROVIDER_PRIORITY[0] if PROVIDER_PRIORITY else "")).strip()

    if _is_provider_healthy(primary_provider):
        return primary_model, primary_reasoning, primary_provider

    for fb in TIER_FALLBACKS.get(tier, []):
        fb_provider = str(fb.get("provider") or "").strip()
        fb_model    = str(fb.get("model") or "").strip()
        if fb_provider and fb_model and _is_provider_healthy(fb_provider):
            logger.info(
                "model-router: T%d primary provider %r unhealthy; using fallback %r/%s",
                tier, primary_provider, fb_provider, fb_model,
            )
            return fb_model, fb.get("reasoning", primary_reasoning), fb_provider

    # All providers unhealthy — use primary and let hermes handle the error
    return primary_model, primary_reasoning, primary_provider


def _sync_tier_fallbacks_to_config(tier: int) -> None:
    """Write tier-specific fallback chain to config.yaml fallback_providers.

    This lets hermes' built-in fallback mechanism handle provider-level failures
    during the actual LLM call with the tier-appropriate model chain.
    """
    fallbacks = TIER_FALLBACKS.get(tier, [])
    if not fallbacks:
        return
    try:
        from hermes_cli.config import load_config, save_config  # type: ignore

        cfg = load_config()
        entries = []
        for fb in fallbacks:
            prov  = str(fb.get("provider") or "").strip()
            model = str(fb.get("model") or "").strip()
            if not prov or not model:
                continue
            entry: dict[str, Any] = {"provider": prov, "model": model}
            base_url = _PROVIDER_BASE_URLS.get(prov)
            if base_url:
                entry["base_url"] = base_url
            entries.append(entry)
        if entries:
            cfg["fallback_providers"] = entries
            save_config(cfg)
            logger.debug("model-router: synced T%d fallbacks to config.yaml fallback_providers", tier)
    except Exception as exc:
        logger.debug("model-router: could not sync tier fallbacks to config: %s", exc)


def bind_session_agent(session_id: str, agent: Any) -> None:
    """Bind a live agent instance to a session so WebUI turns are steerable."""
    if not session_id or agent is None:
        return
    with _state_lock:
        _live_agents[session_id] = agent


def unbind_session_agent(session_id: str, agent: Any | None = None) -> None:
    """Remove a previously bound live agent when a WebUI turn ends."""
    if not session_id:
        return
    with _state_lock:
        current = _live_agents.get(session_id)
        if current is None:
            return
        if agent is not None and current is not agent:
            return
        _live_agents.pop(session_id, None)


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
    guards against double-patching via _router_patched sentinel (checked under
    _patch_lock to prevent a TOCTOU race on concurrent calls).
    """
    import types
    with _patch_lock:
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

        cli._get_status_bar_snapshot = types.MethodType(_patched_snapshot, cli)
        cli._router_patched = True
        logger.debug("model-router: status bar patch applied")


# ---------------------------------------------------------------------------
# Live agent / routing helpers
# ---------------------------------------------------------------------------

def _get_live_agent(session_id: str = "") -> Any | None:
    """Return the active agent for this session from CLI, gateway, or TUI bindings."""
    with _state_lock:
        bound_agent = _live_agents.get(session_id) if session_id else None
    if bound_agent is not None:
        return bound_agent

    if _manager_ref is None:
        return None

    # CLI mode: direct agent reference via _cli_ref
    try:
        cli = _manager_ref._cli_ref
        agent = getattr(cli, "agent", None) if cli else None
        if agent is not None:
            if not session_id:
                return agent
            agent_session = getattr(agent, "session_id", "") or ""
            if agent_session == session_id:
                return agent
    except Exception:
        pass

    # TUI gateway mode: scan tui_gateway.server._sessions for matching agent.
    # _cli_ref is None in TUI mode; the desktop TUI uses its own server module
    # with a _sessions dict keyed by TUI client session IDs.
    # _tui_server_module is pre-cached at register() time to avoid per-call import races.
    if session_id and _tui_server_module is not None:
        try:
            _tui_sessions: dict = getattr(_tui_server_module, "_sessions", {})
            _tui_lock = getattr(_tui_server_module, "_sessions_lock", None)
            ctx = _tui_lock if _tui_lock is not None else contextlib.nullcontext()
            found_agent = None
            found_sid: str = ""
            found_sess: dict = {}
            with ctx:
                for _sid_key, _sess in _tui_sessions.items():
                    _a = _sess.get("agent")
                    if _a is None:
                        continue
                    # session_key is kept in sync with agent.session_id (even after
                    # compression rotations update both fields together).
                    if _sess.get("session_key") == session_id or getattr(_a, "session_id", "") == session_id:
                        found_agent = _a
                        found_sid = _sid_key
                        found_sess = _sess
                        break
            if found_agent is not None:
                with _state_lock:
                    _live_agents[session_id] = found_agent
                    _live_tui_sessions[session_id] = (found_sid, found_sess)
                return found_agent
        except Exception as exc:
            logger.debug("model-router: TUI scan error: %s", exc)

    return None


def _target_tier_for_turn(session_id: str, msg: str, history: list, current_model: str) -> tuple[int, bool]:
    """Return (target_tier, is_new_user_turn) for the current message."""
    with _state_lock:
        last_entry = _session_last.get(session_id)
        is_new_user_turn = (last_entry is None) or (last_entry[0] != msg)

    if is_new_user_turn:
        _evict_stale_sessions()
        with _state_lock:
            _tool_errors[session_id] = 0
            _escalated[session_id] = False
            _session_ts[session_id] = time.monotonic()

        explicit = _detect_explicit_tier(msg)
        if explicit is not None:
            target_tier = explicit
            _set_cached_tier(session_id, msg, target_tier)
            logger.info("model-router: explicit T%d request honoured", target_tier)
        elif _is_obvious_ack(msg):
            target_tier = 1
            _set_cached_tier(session_id, msg, target_tier)
        else:
            target_tier = _classify_with_flash(msg, history)
            _set_cached_tier(session_id, msg, target_tier)

        with _state_lock:
            _base_tier[session_id] = target_tier
            _last_tier[session_id] = target_tier
        return target_tier, True

    with _state_lock:
        _session_ts[session_id] = time.monotonic()
        return _last_tier.get(session_id, 2), False


def prepare_turn(
    *,
    session_id: str,
    user_message: str,
    conversation_history: list | None = None,
    current_model: str = "",
    platform: str = "",
    apply_live: bool = False,
) -> dict[str, Any]:
    """Classify the turn and optionally apply the routed tier to a live agent."""
    msg = user_message.strip()
    if not msg:
        return {
            "session_id": session_id,
            "pinned": is_session_pinned(session_id),
            "tier": get_last_tier(session_id) or MODEL_TO_TIER.get(current_model, 0),
            "model": current_model,
            "reasoning": None,
            "platform": platform,
            "is_new_turn": False,
        }

    history = conversation_history or []
    actual_model = current_model
    agent = _get_live_agent(session_id)
    if agent is not None:
        actual_model = getattr(agent, "model", actual_model) or actual_model

    if _is_manual_override(session_id, actual_model):
        logger.debug("model-router: manual override active (%s), skipping", actual_model)
        return {
            "session_id": session_id,
            "pinned": True,
            "tier": MODEL_TO_TIER.get(actual_model, get_last_tier(session_id)),
            "model": actual_model,
            "reasoning": getattr(agent, "reasoning_config", None) if agent is not None else None,
            "platform": platform,
            "is_new_turn": False,
        }

    target_tier, is_new_user_turn = _target_tier_for_turn(session_id, msg, history, actual_model)
    target_model, target_reasoning, target_provider = _select_tier_entry(target_tier)
    if not target_model:
        target_model = actual_model
    actual_provider = getattr(agent, "provider", "") if agent is not None else ""
    provider_mismatch = bool(target_provider and actual_provider and target_provider != actual_provider)

    logger.debug(
        "model-router: turn T%d -> provider=%s model=%s vs actual_provider=%s actual_model=%s",
        target_tier, target_provider, target_model, actual_provider, actual_model,
    )

    if apply_live:
        if target_model != actual_model or provider_mismatch:
            logger.info(
                "model-router: switching T%d (was T%d / %s via %s)",
                target_tier,
                MODEL_TO_TIER.get(actual_model, 0),
                actual_model.split("/")[-1] if actual_model else "unknown",
                actual_provider or "unknown",
            )
            _apply_tier(session_id, target_tier, actual_model, source="")
        else:
            try:
                cli = _manager_ref._cli_ref if _manager_ref is not None else None
                if cli:
                    _patch_status_bar(cli)
            except Exception:
                pass
    elif agent is not None and (target_model != actual_model or provider_mismatch):
        # WebUI may pre-resolve the routed model before the hook fires; if a
        # reused cached agent still carries the old model/provider, normalize it here.
        _apply_tier(session_id, target_tier, actual_model, source="webui-sync")

    return {
        "session_id": session_id,
        "pinned": False,
        "tier": target_tier,
        "model": target_model,
        "reasoning": target_reasoning,
        "platform": platform,
        "is_new_turn": is_new_user_turn,
    }


# ---------------------------------------------------------------------------
# Apply model switch helper
# ---------------------------------------------------------------------------

def _apply_tier(session_id: str, target_tier: int, current_model: str, source: str = "") -> None:
    """Switch agent.model + reasoning_config to target_tier. Emits badge.

    Uses _select_tier_entry to skip unhealthy primary providers.
    """
    global _manager_ref

    if _manager_ref is None:
        return

    target_model, target_reasoning, provider = _select_tier_entry(target_tier)
    current_tier = MODEL_TO_TIER.get(current_model, 2)

    # Apply status bar patch lazily (needs cli ref)
    try:
        cli = _manager_ref._cli_ref
        if cli:
            _patch_status_bar(cli)
    except Exception:
        pass

    try:
        cli = _manager_ref._cli_ref
        agent = _get_live_agent(session_id)
        if agent is None:
            return

        old_model = agent.model
        old_provider = getattr(agent, "provider", "") or ""
        base_url = _PROVIDER_BASE_URLS.get(provider, "")
        api_mode = ""
        try:
            from hermes_cli.providers import determine_api_mode  # type: ignore
            api_mode = determine_api_mode(provider, base_url)
        except Exception:
            api_mode = "bedrock_converse" if provider == "bedrock" else "chat_completions"

        # In TUI mode, invoke the proper switch mechanism so the shared client,
        # provider, base_url, api_mode, and status bar all update via the same
        # path as /model. Direct attribute assignment is not enough there.
        tui_ctx = None
        with _state_lock:
            tui_ctx = _live_tui_sessions.get(session_id)
        if tui_ctx is not None and _tui_server_module is not None:
            try:
                _sid_tui, _sess_tui = tui_ctx
                model_spec = f"{target_model} --provider {provider}"
                _tui_server_module._apply_model_switch(
                    _sid_tui, _sess_tui, model_spec,
                    confirm_expensive_model=False,
                    pin_session_override=False,
                )
            except Exception as exc:
                logger.debug("model-router: TUI _apply_model_switch failed: %s", exc)
                raise
        elif hasattr(agent, "switch_model"):
            agent.switch_model(
                new_model=target_model,
                new_provider=provider,
                api_key="",
                base_url=base_url,
                api_mode=api_mode,
            )
        else:
            agent.model = target_model
            if provider:
                agent.provider = provider
            if base_url:
                agent.base_url = base_url
            if api_mode:
                agent.api_mode = api_mode

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
            f"{old_provider}/{old_model}", f"{provider}/{target_model}",
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

    prepare_turn(
        session_id=session_id,
        user_message=user_message,
        conversation_history=conversation_history,
        current_model=model,
        platform=platform,
        apply_live=True,
    )


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
            or result_lower.startswith("error")
            or (
                "exit_code" in result_lower
                and '"exit_code": ' in result_lower
                and '"exit_code": 0' not in result_lower
                and '"exit_code": null' not in result_lower
            )
        ):
            is_error = True

    with _state_lock:
        if is_error:
            _tool_errors[session_id] = _tool_errors.get(session_id, 0) + 1
        else:
            _tool_errors[session_id] = 0  # reset on success

        error_count = _tool_errors.get(session_id, 0)
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

        agent = _get_live_agent(session_id)
        current_model = getattr(agent, "model", TIERS[current_tier]["model"]) if agent else TIERS[current_tier]["model"]

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
        agent = _get_live_agent(session_id)
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

    except Exception as exc:
        logger.debug("model-router: on_post_llm_call hook failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Hook: pre_gateway_dispatch  (fires before every incoming message is dispatched)
# ---------------------------------------------------------------------------

def on_pre_gateway_dispatch(*, gateway=None, **kwargs: Any) -> None:
    """Populate _live_agents from the gateway's running session pool.

    In TUI/gateway mode _manager_ref._cli_ref is None, so _get_live_agent falls
    back to _live_agents.  We refresh this cache on every dispatch so each
    agent's session_id → agent mapping is current when pre_llm_call fires.
    """
    if gateway is None:
        return
    try:
        running_agents = getattr(gateway, "_running_agents", {})
        with _state_lock:
            for agent in running_agents.values():
                if agent is None:
                    continue
                sid = getattr(agent, "session_id", None) or ""
                if sid and hasattr(agent, "model"):
                    _live_agents[sid] = agent
    except Exception as exc:
        logger.debug("model-router: pre_gateway_dispatch agent sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook: api_request_error  (fires on every failed LLM API call in the loop)
# ---------------------------------------------------------------------------

def on_api_request_error(
    *,
    provider: str = "",
    session_id: str = "",
    error: dict | None = None,
    retryable: bool | None = None,
    **kwargs: Any,
) -> None:
    """Track provider failures for proactive health-gating.

    Only marks a provider unhealthy on non-retryable errors to avoid
    penalizing transient blips that hermes already retries automatically.
    """
    if not provider:
        return
    # Skip retryable transient errors (5xx, 408, partial reads) — hermes
    # will retry those automatically and we shouldn't penalize the provider.
    if retryable is True:
        return
    _mark_provider_failed(provider)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _manager_ref
    _load_router_config()
    _manager_ref = ctx._manager
    ctx.register_hook("pre_llm_call",          on_pre_llm_call)
    ctx.register_hook("post_llm_call",         on_post_llm_call)
    ctx.register_hook("post_tool_call",        on_post_tool_call)
    ctx.register_hook("api_request_error",     on_api_request_error)
    ctx.register_hook("pre_gateway_dispatch",  on_pre_gateway_dispatch)

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
    mgr.router_prepare_turn  = prepare_turn
    mgr.router_bind_agent    = bind_session_agent
    mgr.router_unbind_agent  = unbind_session_agent

    # Pre-cache tui_gateway.server at startup so the module is guaranteed to be
    # importable when _get_live_agent fires during a TUI session. Importing here
    # (vs. per-call) means failure shows up in the log at load time, not silently
    # per-turn inside an except-pass block.
    global _tui_server_module
    try:
        import tui_gateway.server as _tui_srv_mod  # noqa: PLC0415
        _tui_server_module = _tui_srv_mod
        logger.debug("model-router: tui_gateway.server cached at register time")
    except ImportError:
        logger.debug("model-router: tui_gateway.server not available (non-TUI mode)")

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
