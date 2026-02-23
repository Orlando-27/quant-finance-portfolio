#!/usr/bin/env bash
# Run all 55 CQF modules in order
set -euo pipefail
export MPLBACKEND=Agg
PY="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== CQF Core Modules (51-55) ==="
for f in "$ROOT/src/cqf_core"/cqf_5[1-5]_*.py; do
    echo "  >> $(basename $f)"; "$PY" "$f"
done

echo "=== Thematic Modules (01-50) ==="
for f in "$ROOT/src"/m[0-9][0-9]_*/m[0-9][0-9]_*.py; do
    echo "  >> $(basename $f)"; "$PY" "$f"
done

echo "Done. Figures in outputs/figures/"
