#!/usr/bin/env python3
"""Installer for the model-router plugin.

Goals:
- work for the default Hermes home and all named profiles
- install the plugin in the canonical global location
- symlink named profiles to the canonical plugin dir
- enable the plugin and triage config in each config.yaml
- create skill_routing.md when missing
- append a routing block to SOUL.md when missing
- validate that the local Hermes checkout contains the required CLI patches
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


PLUGIN_NAME = "model-router"
SOUL_BLOCK_START = "<!-- model-router:start -->"
SOUL_BLOCK_END = "<!-- model-router:end -->"
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
            "label": "T2 Plus",
            "emoji": "🔸",
            "model": "qwen/qwen3.6-plus",
            "reasoning": None,
            "role": "default daily-driver",
            "best_for": [
                "Default day-to-day work",
                "Documentation and drafting",
                "Standard coding and research",
            ],
        },
        3: {
            "label": "T3 Haiku",
            "emoji": "🔸🔸",
            "model": "anthropic/claude-haiku-4-5",
            "reasoning": None,
            "role": "stronger fast-path model",
            "best_for": [
                "Debugging",
                "Code review",
                "Large-document synthesis",
            ],
        },
        4: {
            "label": "T4 Sonnet",
            "emoji": "🔶",
            "model": "anthropic/claude-sonnet-4-6",
            "reasoning": "low",
            "role": "deliberate planner",
            "best_for": [
                "Architecture",
                "Migration planning",
                "Complex multi-step design",
            ],
        },
        5: {
            "label": "T5 Sonnet",
            "emoji": "🔶🔶",
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


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def info(msg: str) -> None:
    print(f"  [..] {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def fail(msg: str) -> None:
    print(f"  [fail] {msg}")
    raise SystemExit(1)


def root_home() -> Path:
    """Return the Hermes home root (never a profile subdirectory).

    When HERMES_HOME points at a named profile (e.g. ~/.hermes/profiles/seo),
    we walk up to the real root so canonical plugin dirs and launcher paths
    are resolved correctly.  The named profile is handled by discover_targets().
    """
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        home = Path(env_home).expanduser()
        if home.parent.name == "profiles":
            return home.parent.parent
        return home
    return Path.home() / ".hermes"


def active_profile_from_env() -> str | None:
    """If HERMES_HOME points at a named profile, return its name; else None."""
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        home = Path(env_home).expanduser()
        if home.parent.name == "profiles":
            return home.name
    return None


def canonical_plugin_dir(home_root: Path) -> Path:
    return home_root / "plugins" / PLUGIN_NAME


def plugin_source_dir() -> Path:
    return Path(__file__).resolve().parent


def is_model_router_enabled(home_dir: Path) -> bool:
    cfg = home_dir / "config.yaml"
    if not cfg.exists():
        return False
    text = cfg.read_text(encoding="utf-8")
    return bool(re.search(r"^\s*-\s*model-router\s*$", text, flags=re.MULTILINE))


TIER_COMMANDS_BLOCK = """
    #Model Router
    CommandDef("t1", "Pin session to the configured T1 slot — disables auto-routing until /auto", "Configuration",
               cli_only=True),
    CommandDef("t2", "Pin session to the configured T2 slot — disables auto-routing until /auto", "Configuration",
               cli_only=True),
    CommandDef("t3", "Pin session to the configured T3 slot — disables auto-routing until /auto", "Configuration",
               cli_only=True),
    CommandDef("t4", "Pin session to the configured T4 slot — disables auto-routing until /auto", "Configuration",
               cli_only=True),
    CommandDef("t5", "Pin session to the configured T5 slot — disables auto-routing until /auto", "Configuration",
               cli_only=True),
    CommandDef("auto", "Resume auto model routing (undo /model or /t1-/t5 pin for this session)", "Configuration",
               cli_only=True),
"""

CLI_REF_BLOCK = """        # Ensure plugin manager has a CLI reference even in non-interactive
        # (single-query / -q) mode where run() is never called.
        # This is required for model-router and other plugins that access
        # the agent via ctx._manager._cli_ref.
        try:
            from hermes_cli.plugins import get_plugin_manager
            _pm = get_plugin_manager()
            if _pm._cli_ref is None:
                _pm._cli_ref = self
        except Exception:
            pass

"""

CLI_TIER_HANDLERS_BLOCK = '''    def _handle_tier_pin(self, tier_cmd: str) -> None:
        """/t1 /t2 /t3 /t4 /t5 — pin session to a specific model tier.

        Immediately switches the model and pins auto-routing off for the
        entire session. Use /auto to re-enable auto-routing.
        """
        tier_map = {"t1": 1, "t2": 2, "t3": 3, "t4": 4, "t5": 5}
        tier_num = tier_map.get(tier_cmd.lstrip("/").lower())
        if tier_num is None:
            _cprint(f"  ✗ Unknown tier command: /{tier_cmd}")
            return

        try:
            from hermes_cli.plugins import get_plugin_manager
            _mgr = get_plugin_manager()
            _apply_fn = getattr(_mgr, "router_apply_tier", None)
            _meta_fn = getattr(_mgr, "router_get_tier_meta", None)
            if _apply_fn is None:
                _cprint("  ✗ model-router plugin not active — /t1-/t5 unavailable")
                _cprint("    Enable it: add 'model-router' to plugins.enabled in config.yaml")
                return

            current_model = (
                getattr(self.agent, "model", None) if self.agent else None
            ) or self.model or ""
            session_id = self.session_id or ""

            _apply_fn(session_id, tier_num, current_model)

            # Reflect the tier chosen by the plugin using the active router config
            # instead of baked-in model slugs.
            meta = _meta_fn(tier_num) if _meta_fn else {}
            new_model = meta.get("model")
            if not new_model:
                _cprint(f"  ✗ Tier T{tier_num} is not configured correctly in model_router.yaml")
                _cprint("    Fix the active profile router config or re-run install.sh, then try again.")
                return
            reasoning = meta.get("reasoning")
            tier_label = meta.get("label") or f"T{tier_num}"
            if reasoning:
                tier_label += f" (reasoning={reasoning})"
            if self.agent:
                self.agent.model = new_model
            self.model = new_model

            _cprint(f"  ✓ Pinned to {tier_label} ({new_model})")
            _cprint("    Auto-routing paused for this session. Use /auto to resume.")

        except Exception as exc:
            _cprint(f"  ✗ Failed to switch tier: {exc}")

    def _handle_auto_routing(self) -> None:
        """/auto — resume auto model routing after a /model or /t1-/t5 pin."""
        try:
            from hermes_cli.plugins import get_plugin_manager
            _mgr = get_plugin_manager()
            _unpin_fn = getattr(_mgr, "router_unpin_session", None)
            _is_pinned_fn = getattr(_mgr, "router_is_pinned", None)

            if _unpin_fn is None:
                _cprint("  ✗ model-router plugin not active — /auto unavailable")
                _cprint("    Enable it: add 'model-router' to plugins.enabled in config.yaml")
                return

            session_id = self.session_id or ""
            was_pinned = _is_pinned_fn(session_id) if _is_pinned_fn else False
            _unpin_fn(session_id)

            if was_pinned:
                _cprint("  ✓ Auto model routing resumed")
                _cprint("    Next turn will be classified by Flash and routed automatically.")
            else:
                _cprint("  ✓ Auto routing already active (no pin was set)")

        except Exception as exc:
            _cprint(f"  ✗ Failed to resume auto routing: {exc}")

'''

CLI_PROCESS_COMMAND_BLOCK = """        elif canonical in ("t1", "t2", "t3", "t4", "t5"):
            self._handle_tier_pin(canonical)
        elif canonical == "auto":
            self._handle_auto_routing()
"""

CLI_INLINE_ROUTING_BLOCK = """                # Handle /model directly on the UI thread so interactive pickers
                # can safely use prompt_toolkit terminal handoff helpers.
                if self._should_handle_model_command_inline(text, has_images=has_images):
                    if not self.process_command(text):
                        self._should_exit = True
                        if event.app.is_running:
                            event.app.exit()
                    event.app.current_buffer.reset(append_to_history=True)
                    return

"""

LAUNCHER_TEMPLATE = """#!/usr/bin/env bash
unset PYTHONPATH
unset PYTHONHOME

HERMES_HOME_ROOT="{home_root}"
HERMES_BIN="$HERMES_HOME_ROOT/hermes-agent/venv/bin/hermes"
HERMES_PYTHON="$HERMES_HOME_ROOT/hermes-agent/venv/bin/python"
MODEL_ROUTER_INSTALL="$HERMES_HOME_ROOT/plugins/model-router/install.py"

detect_profile() {{
  local prev=""
  local arg=""
  for arg in "$@"; do
    if [ "$prev" = "-p" ] || [ "$prev" = "--profile" ]; then
      printf '%s' "$arg"
      return 0
    fi
    case "$arg" in
      --profile=*)
        printf '%s' "${{arg#--profile=}}"
        return 0
        ;;
    esac
    prev="$arg"
  done
  printf '%s' "default"
}}

if [ -f "$MODEL_ROUTER_INSTALL" ] && [ -x "$HERMES_PYTHON" ]; then
  PROFILE_NAME="$(detect_profile "$@")"
  STARTUP_STATUS="$("$HERMES_PYTHON" "$MODEL_ROUTER_INSTALL" --startup-status "$PROFILE_NAME" 2>/dev/null || true)"
  if [ "$STARTUP_STATUS" = "invalid" ]; then
    if [ -t 0 ] && [ -t 1 ]; then
      printf '%s' "Hermes model-router startup validation failed. Plugin model-router won't work correctly without repair. Do you want to run install.sh repair now? [y/N] "
      read -r reply
      case "$reply" in
        y|Y|yes|YES)
          if ! "$HERMES_PYTHON" "$MODEL_ROUTER_INSTALL" "$PROFILE_NAME"; then
            printf '%s\\n' "model-router repair failed."
            exit 1
          fi
          STARTUP_STATUS="$("$HERMES_PYTHON" "$MODEL_ROUTER_INSTALL" --startup-status "$PROFILE_NAME" 2>/dev/null || true)"
          if [ "$STARTUP_STATUS" != "ok" ]; then
            printf '%s\\n' "Hermes model-router startup validation still failed after repair."
            exit 1
          fi
          ;;
        *)
          printf '%s\\n' "Continuing without repair. model-router may not work correctly."
          ;;
      esac
    else
      printf '%s\\n' "Hermes model-router startup validation failed and no interactive terminal is available for repair prompt." >&2
    fi
  fi
fi

exec "$HERMES_BIN" "$@"
"""


def _write_with_backup_if_changed(path: Path, new_text: str) -> bool:
    old_text = path.read_text(encoding="utf-8")
    if old_text == new_text:
        return False
    backup = backup_path(path)
    shutil.copy2(path, backup)
    path.write_text(new_text, encoding="utf-8")
    ok(f"Patched {path.name} (backup: {backup.name})")
    return True


def _replace_or_insert_block(
    text: str,
    expected_block: str,
    existing_pattern: str,
    insert_anchor_pattern: str,
    insert_after_match: bool = True,
    missing_error: str = "",
) -> str:
    if expected_block in text:
        return text

    if existing_pattern:
        text = re.sub(existing_pattern, "", text, count=1, flags=re.MULTILINE | re.DOTALL)

    anchor_match = re.search(insert_anchor_pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not anchor_match:
        fail(missing_error or f"Could not locate insertion anchor: {insert_anchor_pattern}")

    insert_at = anchor_match.end() if insert_after_match else anchor_match.start()
    return text[:insert_at] + expected_block + text[insert_at:]


def repair_commands_py(commands_path: Path) -> bool:
    if not commands_path.exists():
        return False
    text = commands_path.read_text(encoding="utf-8")
    cleaned = re.sub(
        r"^\s*#Model Router\s*\n(?:^\s*$\n|^\s*#Model Router\s*\n)*",
        "",
        text,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(
        r'^\s*CommandDef\("t1".*?^\s*CommandDef\("auto".*?^\s*cli_only=True\),\n?',
        "",
        cleaned,
        flags=re.MULTILINE | re.DOTALL,
    )
    cleaned = _replace_or_insert_block(
        text=cleaned,
        expected_block=TIER_COMMANDS_BLOCK,
        existing_pattern="",
        insert_anchor_pattern=r"^\]$",
        insert_after_match=False,
        missing_error="Could not locate command registry closing bracket in commands.py for model-router patching",
    )
    return _write_with_backup_if_changed(commands_path, cleaned)


def repair_cli_py(cli_path: Path) -> bool:
    if not cli_path.exists():
        return False

    text = cli_path.read_text(encoding="utf-8")
    original = text

    text = _replace_or_insert_block(
        text=text,
        expected_block=CLI_REF_BLOCK,
        existing_pattern=r'^\s*# Ensure plugin manager has a CLI reference even in non-interactive.*?^\s*pass\n+',
        insert_anchor_pattern=r"^\s*if not self\._ensure_runtime_credentials\(\):\n\s*return False\n",
        insert_after_match=True,
        missing_error="Could not locate runtime credential guard in cli.py for model-router patching",
    )

    text = _replace_or_insert_block(
        text=text,
        expected_block=CLI_TIER_HANDLERS_BLOCK,
        existing_pattern=r'^\s*def _handle_tier_pin\(self, tier_cmd: str\) -> None:.*?(?=^\s*def _should_handle_model_command_inline\()',
        insert_anchor_pattern=r'^\s*def _should_handle_model_command_inline\(',
        insert_after_match=False,
        missing_error="Could not locate _should_handle_model_command_inline anchor in cli.py for model-router patching",
    )

    text = _replace_or_insert_block(
        text=text,
        expected_block=CLI_PROCESS_COMMAND_BLOCK,
        existing_pattern=r'^\s*elif canonical in \("t1", "t2", "t3", "t4", "t5"\):\n\s*self\._handle_tier_pin\(canonical\)\n\s*elif canonical == "auto":\n\s*self\._handle_auto_routing\(\)\n',
        insert_anchor_pattern=r'^\s*elif canonical == "model":\n\s*self\._handle_model_switch\(cmd_original\)\n',
        insert_after_match=True,
        missing_error="Could not locate /model branch in cli.py for model-router patching",
    )

    text = _replace_or_insert_block(
        text=text,
        expected_block=CLI_INLINE_ROUTING_BLOCK,
        existing_pattern=r'^\s*# Handle /model directly on the UI thread so interactive pickers.*?^\s*return\n\n',
        insert_anchor_pattern=r"^\s*if text or has_images:\n",
        insert_after_match=True,
        missing_error="Could not locate input routing block in cli.py for model-router patching",
    )

    if text == original:
        return False
    return _write_with_backup_if_changed(cli_path, text)


def repair_hermes_core(home_root: Path) -> None:
    repo = home_root / "hermes-agent"
    cli_path = repo / "cli.py"
    commands_path = repo / "hermes_cli" / "commands.py"
    if not cli_path.exists() or not commands_path.exists():
        warn("Hermes source checkout not found; skipping core repair")
        return

    changed = False
    changed = repair_commands_py(commands_path) or changed
    changed = repair_cli_py(cli_path) or changed
    if not changed:
        ok("Hermes core integrations already patched")


def collect_missing_core_integrations(home_root: Path) -> list[str]:
    repo = home_root / "hermes-agent"
    cli_path = repo / "cli.py"
    commands_path = repo / "hermes_cli" / "commands.py"
    if not cli_path.exists() or not commands_path.exists():
        return []

    cli_text = cli_path.read_text(encoding="utf-8")
    commands_text = commands_path.read_text(encoding="utf-8")

    missing: list[str] = []
    if f"{TIER_COMMANDS_BLOCK}]" not in commands_text:
        missing.append("complete slash command block for /t1-/t5 and /auto in the command registry")
    if CLI_REF_BLOCK not in cli_text:
        missing.append("complete cli.py _cli_ref block for non-interactive mode")
    if CLI_TIER_HANDLERS_BLOCK not in cli_text:
        missing.append("complete CLI tier handler block for /t1-/t5 and /auto")
    if CLI_PROCESS_COMMAND_BLOCK not in cli_text:
        missing.append("complete process_command dispatch block for /t1-/t5 and /auto")
    if CLI_INLINE_ROUTING_BLOCK not in cli_text:
        missing.append("complete inline UI-thread routing block for /model and /t1-/t5")
    return missing


def compat_check(home_root: Path) -> None:
    repo = home_root / "hermes-agent"
    cli_path = repo / "cli.py"
    commands_path = repo / "hermes_cli" / "commands.py"
    if not cli_path.exists() or not commands_path.exists():
        warn("Hermes source checkout not found; skipping compatibility checks")
        return

    missing = collect_missing_core_integrations(home_root)

    if missing:
        fail(
            "Hermes checkout is missing required model-router integration:\n"
            + "\n".join(f"    - {item}" for item in missing)
        )

    ok("Compatibility check passed")


def sync_global_plugin(home_root: Path) -> Path:
    src = plugin_source_dir()
    dst = canonical_plugin_dir(home_root)

    if src.resolve() == dst.resolve():
        ok(f"Canonical plugin source already in place: {dst}")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    ok(f"Synced plugin source to canonical location: {dst}")
    return dst


def ensure_global_launcher(home_root: Path) -> None:
    default_home = Path.home() / ".hermes"
    if home_root != default_home:
        info("Skipping launcher patch because HERMES_HOME is not the default ~/.hermes")
        return

    launcher_path = Path.home() / ".local" / "bin" / "hermes"
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    new_text = LAUNCHER_TEMPLATE.format(home_root=str(home_root))
    if launcher_path.exists():
        old_text = launcher_path.read_text(encoding="utf-8")
        if old_text == new_text:
            ok("Global hermes launcher already current")
            return
        backup = backup_path(launcher_path)
        shutil.copy2(launcher_path, backup)
        launcher_path.write_text(new_text, encoding="utf-8")
        launcher_path.chmod(0o755)
        ok(f"Updated global hermes launcher (backup: {backup.name})")
        return

    launcher_path.write_text(new_text, encoding="utf-8")
    launcher_path.chmod(0o755)
    ok("Created global hermes launcher with model-router startup guard")


def discover_targets(home_root: Path, argv: list[str]) -> list[tuple[str, Path]]:
    """Return (name, path) pairs for every target to install into.

    Special case: when HERMES_HOME=~/.hermes/profiles/<name> (i.e. the caller
    is already running *inside* a named profile), active_profile_from_env()
    returns that profile name.  Without this check, discover_targets() would
    only add the profile as part of the auto-scan — but the auto-scan walks
    home_root/profiles/, and home_root is the *root* (~/.hermes), so the
    named profile *is* discovered correctly in the no-argv branch.

    The real bug was subtler: install.sh was invoked with HERMES_HOME set to
    the profile dir, root_home() stripped the profile suffix, and then
    ensure_profile_plugin_link() skipped name=="default" — so the active
    named profile never got its symlink.  We fix this by ensuring that when
    HERMES_HOME points at a named profile and no explicit argv is given, that
    profile appears in the target list under its real name (not "default").
    """
    if argv:
        targets: list[tuple[str, Path]] = []
        for name in argv:
            canon = name.strip()
            if not canon:
                continue
            if canon == "default":
                targets.append(("default", home_root))
            else:
                targets.append((canon, home_root / "profiles" / canon))
        return targets

    # No explicit targets — auto-discover.
    targets: list[tuple[str, Path]] = [("default", home_root)]
    profiles_dir = home_root / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if child.is_dir() and (child / "config.yaml").exists():
                targets.append((child.name, child))

    # If the caller is running inside a named profile (HERMES_HOME=.../profiles/<name>)
    # and that profile was NOT found in the auto-scan (e.g. its config.yaml lives
    # somewhere unexpected), add it explicitly so its symlink is always created.
    active = active_profile_from_env()
    if active and active != "default":
        already = any(name == active for name, _ in targets)
        if not already:
            profile_path = home_root / "profiles" / active
            if profile_path.is_dir():
                targets.append((active, profile_path))
                info(f"Added active profile '{active}' from HERMES_HOME to install targets")

    return targets


def backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}.bak.model-router.{stamp}")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_router_config(raw: dict | None) -> dict:
    merged = _deep_merge(DEFAULT_ROUTER_CONFIG, raw or {})
    normalized_tiers: dict[int, dict] = {}
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


def router_config_path(home_dir: Path) -> Path:
    return home_dir / "model_router.yaml"


def render_router_config(router_config: dict) -> str:
    return yaml.safe_dump(
        router_config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def ensure_router_config(home_dir: Path) -> dict:
    path = router_config_path(home_dir)
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError("model_router.yaml must be a mapping")
            config = normalize_router_config(raw)
        except Exception as exc:
            fail(f"{path} is invalid: {exc}")
    else:
        config = normalize_router_config({})

    rendered = render_router_config(config)
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing != rendered:
        path.write_text(rendered, encoding="utf-8")
        action = "created" if existing is None else "normalized"
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: {action} model_router.yaml")
    else:
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: model_router.yaml already current")
    return config


def ensure_profile_plugin_link(name: str, home_dir: Path, canonical_dir: Path) -> None:
    if name == "default":
        return

    plugins_dir = home_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    link = plugins_dir / PLUGIN_NAME

    if link.is_symlink():
        target = link.resolve()
        if target == canonical_dir.resolve():
            ok(f"{name}: plugin symlink already points to canonical source")
            return
        link.unlink()
        link.symlink_to(canonical_dir)
        ok(f"{name}: updated plugin symlink -> {canonical_dir}")
        return

    if link.exists():
        backup = backup_path(link)
        shutil.move(str(link), str(backup))
        warn(f"{name}: existing plugin dir/file moved to {backup}")

    link.symlink_to(canonical_dir)
    ok(f"{name}: created plugin symlink -> {canonical_dir}")


def ensure_plugin_enabled(text: str) -> tuple[str, bool]:
    if re.search(r"^\s*-\s*model-router\s*$", text, flags=re.MULTILINE):
        return text, False

    plugins_match = re.search(r"^plugins:\n", text, flags=re.MULTILINE)
    if not plugins_match:
        return text.rstrip("\n") + "\nplugins:\n  enabled:\n    - model-router\n", True

    enabled_match = re.search(r"^plugins:\n(?:  .*\n)*?  enabled:\n", text, flags=re.MULTILINE)
    if enabled_match:
        insert_at = enabled_match.end()
        return text[:insert_at] + "    - model-router\n" + text[insert_at:], True

    insert_at = plugins_match.end()
    return text[:insert_at] + "  enabled:\n    - model-router\n" + text[insert_at:], True


def render_triage_specifier_block(router_config: dict) -> str:
    classifier = router_config["classifier"]
    lines = [
        "  triage_specifier:",
        f"    provider: {classifier['provider']}",
        f"    model: {classifier['model']}",
        f"    base_url: {classifier['base_url']}",
        "    api_key: ''",
        f"    timeout: {classifier['timeout']}",
        "    extra_body:",
    ]
    extra_body = classifier.get("extra_body", {})
    if extra_body:
        for key, value in extra_body.items():
            rendered = "true" if value is True else "false" if value is False else value
            lines.append(f"      {key}: {rendered}")
    else:
        lines.append("      {}")
    return "\n".join(lines) + "\n"


def ensure_triage_specifier(text: str, router_config: dict) -> tuple[str, bool]:
    block = render_triage_specifier_block(router_config)
    pattern = r"^  triage_specifier:\n(?:    .*\n|      .*\n)*"
    if re.search(pattern, text, flags=re.MULTILINE):
        updated = re.sub(pattern, block, text, count=1, flags=re.MULTILINE)
        return updated, updated != text

    aux_match = re.search(r"^auxiliary:\n", text, flags=re.MULTILINE)
    if not aux_match:
        return text.rstrip("\n") + "\nauxiliary:\n" + block, True

    insert_at = aux_match.end()
    return text[:insert_at] + block + text[insert_at:], True


def ensure_config(home_dir: Path, router_config: dict) -> None:
    cfg = home_dir / "config.yaml"
    if not cfg.exists():
        warn(f"{home_dir}: missing config.yaml, skipping config patch")
        return

    text = cfg.read_text(encoding="utf-8")
    changed = False
    text, delta = ensure_plugin_enabled(text)
    changed = changed or delta
    text, delta = ensure_triage_specifier(text, router_config)
    changed = changed or delta
    if changed:
        cfg.write_text(text, encoding="utf-8")
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: updated config.yaml")
    else:
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: config.yaml already configured")


def render_skill_routing(home_dir: Path, router_config: dict) -> str:
    classifier = router_config["classifier"]
    lines = [
        "# Hermes Model Routing Policy\n\n"
        "This file documents the cost-aware routing tiers used by the model-router plugin.\n"
        "Routing is AUTOMATIC -- the plugin classifies each turn by complexity and switches\n"
        "the model before every LLM call. No manual /model switching needed.\n\n"
        "Plugin location: active Hermes home `plugins/model-router`\n"
        "Router config:    active Hermes home `model_router.yaml`\n"
        f"Classifier:       {classifier['provider']} / {classifier['model']}\n\n"
        "## Goals\n\n"
        "- Default to the cheapest model that can likely complete the task well.\n"
        "- Escalate only when the task actually needs deeper reasoning or better context handling.\n"
        "- Keep expensive Sonnet usage rare and intentional.\n\n"
        "## Tier Table\n\n"
    ]
    for tier_num in range(1, 6):
        meta = router_config["tiers"][tier_num]
        lines.append(f"Tier {tier_num}\n")
        lines.append(f"  Label:     {meta['label']}\n")
        lines.append(f"  Model:     {meta['model']}\n")
        if meta.get("reasoning"):
            lines.append(f"  Reasoning: {meta['reasoning']}\n")
        lines.append(f"  Role:      {meta.get('role', '')}\n")
        lines.append("  Triggers:\n")
        for item in meta.get("best_for", []):
            lines.append(f"    - {item}\n")
        lines.append("\n")

    lines.extend(
        [
            "## Escalation Rules\n\n",
            "1. Start with Tier 2 for most normal user work.\n",
            "2. Use Tier 1 for ultra-short acks, triage, or mechanical helper tasks.\n",
            "3. Escalate to Tier 3 when debugging, reviewing, or handling large docs.\n",
            "4. Escalate to Tier 4 for design, planning, or architecture work.\n",
            "5. Escalate to Tier 5 only for security-critical or algorithmically dense tasks.\n",
            "6. Fail fast: if unsure, pick the cheaper tier first.\n\n",
            "## Cost Notes\n\n",
            "- Prompt caching should remain enabled.\n",
            "- Keep cheap models on auxiliary tasks.\n",
            "- Avoid using Sonnet for rote edits, summaries, or shell/file boilerplate.\n",
        ]
    )
    return "".join(lines)


def ensure_skill_routing(home_dir: Path, router_config: dict) -> None:
    path = home_dir / "skill_routing.md"
    rendered = render_skill_routing(home_dir, router_config)
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing != rendered:
        path.write_text(rendered, encoding="utf-8")
        action = "created" if existing is None else "updated"
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: {action} skill_routing.md")
    else:
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: skill_routing.md already current")


def render_soul_block(home_dir: Path) -> str:
    skill_path = home_dir / "skill_routing.md"
    return (
        f"{SOUL_BLOCK_START}\n"
        "When multiple model tiers are available, prefer the cheapest tier that is likely to succeed.\n"
        "Default to the day-to-day model first, then escalate only when the task truly needs deeper reasoning, better long-context handling, or higher confidence.\n\n"
        f"Use `{skill_path}` as the shared local routing policy for model escalation.\n\n"
        "Routing posture:\n"
        "- Keep cheap helper tasks on the lightest tier.\n"
        "- Use the default tier for most ordinary work.\n"
        "- Escalate to stronger tiers for debugging, architecture, security, and unusually complex reasoning.\n"
        "- If uncertain between two tiers, start with the cheaper one and escalate only after weak output or clear mismatch.\n"
        "- Apply fail-fast routing: if task complexity is unclear, choose the lowest plausible tier first.\n"
        "- Treat one cheap failed attempt as acceptable; escalate after mismatch, weak output, or repeated failure.\n"
        f"{SOUL_BLOCK_END}\n"
    )


def ensure_soul(home_dir: Path) -> None:
    path = home_dir / "SOUL.md"
    block = render_soul_block(home_dir)
    if not path.exists():
        path.write_text("# Hermes Agent Persona\n\n" + block, encoding="utf-8")
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: created SOUL.md")
        return

    text = path.read_text(encoding="utf-8")
    if SOUL_BLOCK_START in text and SOUL_BLOCK_END in text:
        new_text = re.sub(
            re.escape(SOUL_BLOCK_START) + r".*?" + re.escape(SOUL_BLOCK_END) + r"\n?",
            block,
            text,
            flags=re.DOTALL,
        )
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: refreshed router block in SOUL.md")
        else:
            ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: SOUL.md router block already current")
        return

    legacy_pattern = (
        r"When multiple model tiers are available, prefer the cheapest tier that is likely to succeed\.\n"
        r".*?"
        r"- Treat one cheap failed attempt as acceptable; escalate after mismatch, weak output, or repeated failure\.\n?"
    )
    if re.search(legacy_pattern, text, flags=re.DOTALL):
        new_text = re.sub(legacy_pattern, block, text, count=1, flags=re.DOTALL)
        path.write_text(new_text, encoding="utf-8")
        ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: replaced legacy routing block in SOUL.md")
        return

    sep = "" if text.endswith("\n\n") else "\n\n"
    path.write_text(text + sep + block, encoding="utf-8")
    ok(f"{home_dir.name if home_dir.name != '.hermes' else 'default'}: appended router block to SOUL.md")


def startup_status(home_root: Path, profile_name: str) -> str:
    target_home = home_root if profile_name == "default" else home_root / "profiles" / profile_name
    if not is_model_router_enabled(target_home):
        return "disabled"
    if not router_config_path(target_home).exists():
        return "invalid"
    # Also check that the profile symlink exists (can be missing if the profile
    # was created after install.sh ran — the auto-scan skips profiles without
    # config.yaml at install time, so a late-added profile never gets its link).
    if profile_name != "default":
        link = target_home / "plugins" / PLUGIN_NAME
        if not link.exists():
            return "invalid"
    return "invalid" if collect_missing_core_integrations(home_root) else "ok"


def install_for_target(name: str, home_dir: Path, canonical_dir: Path) -> None:
    if not home_dir.exists():
        warn(f"{name}: target home does not exist at {home_dir}, skipping")
        return

    print(f"\n=== Target: {name} ({home_dir}) ===")
    router_config = ensure_router_config(home_dir)
    ensure_profile_plugin_link(name, home_dir, canonical_dir)
    ensure_config(home_dir, router_config)
    ensure_skill_routing(home_dir, router_config)
    ensure_soul(home_dir)


def main(argv: list[str]) -> int:
    home_root = root_home()
    if argv and argv[0] == "--startup-status":
        profile_name = argv[1] if len(argv) > 1 and argv[1].strip() else "default"
        print(startup_status(home_root, profile_name))
        return 0

    repair_hermes_core(home_root)
    compat_check(home_root)
    canonical_dir = sync_global_plugin(home_root)
    ensure_global_launcher(home_root)
    targets = discover_targets(home_root, argv)

    if not targets:
        warn("No install targets found")
        return 0

    info("Targets: " + ", ".join(name for name, _ in targets))
    for name, home_dir in targets:
        install_for_target(name, home_dir, canonical_dir)

    print("\nDone. Restart Hermes for changes to take effect.")
    print("Verify with: hermes plugins list")
    print("Named profiles use: <profile> plugins list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
