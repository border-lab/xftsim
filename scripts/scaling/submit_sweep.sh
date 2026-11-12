#!/bin/bash
# Submit one Slurm job per row of grid.tsv (run setup_lane.sh first).
#
#     bash submit_sweep.sh [grid.tsv]
#     PARTITION=<name> ACCOUNT=<name> bash submit_sweep.sh     # if Lane needs them
#
# Results and logs go to $OUT (default <base>/results/<timestamp>).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=${BASE:-$(dirname "$(dirname "$REPO")")}
GRID=${1:-$HERE/grid.tsv}
OUT=${OUT:-$BASE/results/$(date +%Y%m%d_%H%M)}
[ -f "$BASE/activate.sh" ] || { echo "run setup_lane.sh first"; exit 1; }
mkdir -p "$OUT/logs"

grep -v '^#' "$GRID" | grep -v '^[[:space:]]*$' |
while IFS=$'\t' read -r name people chrom left length gens mem walltime; do
    job=$(sbatch --parsable --job-name="xft_$name" --cpus-per-task=2 \
        --mem="$mem" --time="$walltime" \
        ${PARTITION:+--partition="$PARTITION"} ${ACCOUNT:+--account="$ACCOUNT"} \
        --output="$OUT/logs/$name.%j.log" \
        --wrap="source '$BASE/activate.sh' && python -u '$HERE/scaling_run.py' --name '$name' --people $people --chrom $chrom --left-mb $left --length-mb $length --gens $gens --out '$OUT'" < /dev/null)
    printf '%-16s job %-10s mem %-5s time %s\n' "$name" "$job" "$mem" "$walltime"
done
echo "results -> $OUT"
echo "status:  squeue -u \$USER     summary:  python $HERE/summarize.py $OUT"
