#!/usr/bin/env bash
# Thin wrapper around the Python installer so the plugin can do profile-aware,
# idempotent setup without brittle shell regex hacks.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$PLUGIN_DIR/install.py" "$@"
