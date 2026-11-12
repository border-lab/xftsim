#!/bin/bash
# Laptop backup: run the laptop-sized rows of grid.tsv one after another.
#
#     caffeinate -i bash scripts/scaling/run_local.sh      # caffeinate keeps the Mac awake
#
# A failed run is recorded and the next one starts.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$REPO/.venv-xftsim/bin/python}
export PATH="$(dirname "$PY"):$PATH"   # founders call the venv's `grg` CLI
OUT=${OUT:-$REPO/scaling_results/local_$(date +%Y%m%d_%H%M)}
GRID=${GRID:-$HERE/grid.tsv}
RUNS=${RUNS:-"n050k_L20 n100k_L20 n500k_L5 n250k_L20 n500k_L20 n100k_L20_g50 n1M_L20 n500k_chr22 n500k_L20_g30"}
mkdir -p "$OUT/logs"

for run in $RUNS; do
    row=$(awk -F'\t' -v r="$run" '$1 == r' "$GRID")
    [ -n "$row" ] || { echo "no row $run in $GRID"; continue; }
    IFS=$'\t' read -r name people chrom left length gens _ <<< "$row"
    echo "$(date +%H:%M) start $name"
    "$PY" -u "$HERE/scaling_run.py" --name "$name" --people "$people" --chrom "$chrom" \
        --left-mb "$left" --length-mb "$length" --gens "$gens" --out "$OUT" \
        > "$OUT/logs/$name.log" 2>&1
    rc=$?
    echo "$(date +%H:%M) end   $name (exit $rc)"
done
"$PY" "$HERE/summarize.py" "$OUT"
