#!/usr/bin/env bash
set -euo pipefail
BASE="$(pwd)"

echo ""
echo "================================================================="
echo "  AUDIT: 19-cqf-concepts-explained"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================="

echo ""
echo "--- MODULOS EN DISCO (src/) ---"
echo ""
printf "%-8s %-50s %-10s %s\n" "MOD" "SCRIPT" "LINES" "FIGS"
echo "------------------------------------------------------------------------"

total_scripts=0
total_figs=0

for dir in "$BASE/src"/m*/; do
    mod=$(basename "$dir")
    script=$(find "$dir" -name "*.py" 2>/dev/null | head -1)
    if [ -z "$script" ]; then
        printf "%-8s %-50s %-10s %s\n" "$mod" "(NO SCRIPT)" "-" "-"
        continue
    fi
    fname=$(basename "$script")
    lines=$(wc -l < "$script")
    fig_dir="$BASE/figures/${mod}"
    if [ -d "$fig_dir" ]; then
        n_figs=$(find "$fig_dir" -name "*.png" 2>/dev/null | wc -l)
    else
        n_figs=0
    fi
    printf "%-8s %-50s %-10s %d figs\n" "$mod" "$fname" "${lines}L" "$n_figs"
    total_scripts=$((total_scripts + 1))
    total_figs=$((total_figs + n_figs))
done

echo "------------------------------------------------------------------------"
echo "  Total scripts encontrados : $total_scripts"
echo "  Total figuras generadas   : $total_figs"

echo ""
echo "--- POSIBLES DUPLICADOS (keywords en nombre de archivo) ---"
echo ""
for kw in nlp sentiment var cvar hmm regime momentum neural lstm rnn backtest factor; do
    matches=$(find "$BASE/src" -name "*${kw}*" -name "*.py" 2>/dev/null | sort)
    count=$(echo "$matches" | grep -c . 2>/dev/null || true)
    if [ "${count:-0}" -gt 1 ]; then
        echo "  '$kw' aparece $count veces:"
        echo "$matches" | while read -r f; do
            echo "      $(basename $(dirname $f))/$(basename $f)"
        done
    fi
done

echo ""
echo "--- SCRIPTS SIN FIGURAS (posibles fallos) ---"
echo ""
empty=0
for dir in "$BASE/src"/m*/; do
    mod=$(basename "$dir")
    fig_dir="$BASE/figures/${mod}"
    script=$(find "$dir" -name "*.py" 2>/dev/null | head -1)
    [ -z "$script" ] && continue
    n=0
    [ -d "$fig_dir" ] && n=$(find "$fig_dir" -name "*.png" 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  SIN FIGS: $mod"
        empty=$((empty + 1))
    fi
done
[ "$empty" -eq 0 ] && echo "  Todos los scripts tienen figuras."

echo ""
echo "--- SCRIPTS CORTOS < 100 lineas (posibles stubs) ---"
echo ""
stubs=0
for dir in "$BASE/src"/m*/; do
    script=$(find "$dir" -name "*.py" 2>/dev/null | head -1)
    [ -z "$script" ] && continue
    lines=$(wc -l < "$script")
    if [ "$lines" -lt 100 ]; then
        echo "  STUB: $(basename $(dirname $script))/$(basename $script)  (${lines}L)"
        stubs=$((stubs + 1))
    fi
done
[ "$stubs" -eq 0 ] && echo "  No hay stubs detectados."

echo ""
echo "--- DIRECTORIOS PRESENTES ---"
echo ""
find "$BASE/src" -maxdepth 1 -type d -name "m*" | sort | xargs -I{} basename {}

echo ""
echo "================================================================="
echo "  AUDIT COMPLETE"
echo "================================================================="
