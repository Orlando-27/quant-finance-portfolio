#!/usr/bin/env bash
# =============================================================================
# PROJECT 20: FINANCIAL MODELING TEMPLATES -- RUN ALL MODULES
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
export MPLBACKEND=Agg

echo "============================================================"
echo "  PROJECT 20: FINANCIAL MODELING TEMPLATES"
echo "  Running all 6 modules..."
echo "============================================================"

MODULES=(
    "src/dcf_valuation.py"
    "src/lbo_model.py"
    "src/three_statement_model.py"
    "src/comparable_analysis.py"
    "src/merger_model.py"
    "src/sensitivity_analysis.py"
)

PASSED=0
FAILED=0

for module in "${MODULES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Running: $module"
    echo "------------------------------------------------------------"
    if "$PYTHON" "$ROOT/$module"; then
        echo "  [PASS] $module"
        ((PASSED++))
    else
        echo "  [FAIL] $module"
        ((FAILED++))
    fi
done

echo ""
echo "============================================================"
echo "  RESULTS: $PASSED passed, $FAILED failed"
echo "  Figures: $(find "$ROOT/outputs/figures" -name '*.png' | wc -l) PNG files"
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
