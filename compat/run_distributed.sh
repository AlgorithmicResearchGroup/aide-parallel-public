#!/usr/bin/env bash
# Deprecated wrapper: use ./cli/aide-run

echo "[DEPRECATED] run_distributed.sh is deprecated. Use ./cli/aide-run instead." >&2
exec "$(cd "$(dirname "$0")/.." && pwd)/cli/aide-run" "$@"
