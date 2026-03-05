#!/usr/bin/env bash
# Deprecated wrapper: use ./aide-cluster-down

echo "[DEPRECATED] stop_cluster.sh is deprecated. Use ./aide-cluster-down instead." >&2
exec "$(cd "$(dirname "$0")/.." && pwd)/cli/aide-cluster-down" "$@"
