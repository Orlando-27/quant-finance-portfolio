#!/usr/bin/env bash
set -euo pipefail
BASE="$(pwd)"

echo "================================================================="
echo "  FIX FIGURES -- Unificar todas las figuras en figures/"
echo "================================================================="

# -----------------------------------------------------------------------
# 1. MOVER figuras existentes en outputs/figures/ a figures/
# -----------------------------------------------------------------------
echo ""
echo "--- MOVIENDO outputs/figures/ -> figures/ ---"

# M03-M21: archivos planos outputs/figures/mXX_*.png
for mod in $(seq -w 3 21); do
    dest="$BASE/figures/m${mod}"
    mkdir -p "$dest"
    count=0
    for f in "$BASE/outputs/figures/m${mod}_"*.png 2>/dev/null; do
        [ -f "$f" ] || continue
        cp "$f" "$dest/"
        count=$((count+1))
    done
    [ "$count" -gt 0 ] && echo "  m${mod}: $count figs moved"
done

# M43-M51: subdirectorios outputs/figures/m4X/
for mod in 43 44 45 46 47 48 49 50 51; do
    src_dir="$BASE/outputs/figures/m${mod}"
    dest="$BASE/figures/m${mod}"
    if [ -d "$src_dir" ]; then
        mkdir -p "$dest"
        count=$(find "$src_dir" -name "*.png" | wc -l)
        cp "$src_dir/"*.png "$dest/" 2>/dev/null || true
        echo "  m${mod}: $count figs moved from outputs/figures/m${mod}/"
    else
        echo "  m${mod}: outputs/figures/m${mod}/ NOT FOUND"
    fi
done

# -----------------------------------------------------------------------
# 2. INVENTARIO de lo que hay en figures/ ahora
# -----------------------------------------------------------------------
echo ""
echo "--- INVENTARIO figures/ ---"
total_figs=0
for i in $(seq -w 1 56); do
    dir=$(find "$BASE/src" -maxdepth 1 -type d -name "m${i}_*" 2>/dev/null | head -1)
    [ -z "$dir" ] && continue
    mod_name=$(basename "$dir")
    # Check both naming conventions
    n1=$(find "$BASE/figures/m${i}" -name "*.png" 2>/dev/null | wc -l)
    n2=$(find "$BASE/figures/${mod_name}" -name "*.png" 2>/dev/null | wc -l)
    n=$((n1 + n2))
    if [ "$n" -gt 0 ]; then
        echo "  m${i} ($mod_name): $n figs -- OK"
    else
        echo "  m${i} ($mod_name): 0 figs -- NEEDS RUN"
    fi
    total_figs=$((total_figs + n))
done
echo ""
echo "  Total figuras en figures/: $total_figs"

# -----------------------------------------------------------------------
# 3. DETECTAR scripts que usan paths hardcoded con ~
# -----------------------------------------------------------------------
echo ""
echo "--- SCRIPTS CON PATH HARDCODED (~/quant...) ---"
grep -rl "~/quant-finance-portfolio" "$BASE/src" 2>/dev/null | while read f; do
    echo "  HARDCODED: $f"
done

echo ""
echo "================================================================="
echo "  FIX COMPLETE -- Ver modulos con 0 figs arriba para re-correr"
echo "================================================================="
