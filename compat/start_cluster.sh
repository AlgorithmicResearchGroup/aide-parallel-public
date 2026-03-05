#!/usr/bin/env bash
# Deprecated wrapper: use ./aide-cluster-up

echo "[DEPRECATED] start_cluster.sh is deprecated. Use ./aide-cluster-up instead." >&2
exec "$(cd "$(dirname "$0")/.." && pwd)/cli/aide-cluster-up" "$@"
