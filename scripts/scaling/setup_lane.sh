#!/bin/bash
# Build the xftsim + GRG environment on a Linux cluster, then check it.
#
# Run from the unpacked bundle (layout: <base>/src/{xftsim,grgl,stdpopsim_cache}):
#     bash <base>/src/xftsim/scripts/scaling/setup_lane.sh
# Everything is installed under <base> (override with BASE=...): a private
# Miniforge, the conda env, and the stdpopsim genetic-map cache. Needs internet
# (conda and pip downloads), so run it on a login node. Safe to re-run.
#
# pygrgl is built from the bundled grgl source (branch grg_modify_improve,
# commit 8e716cb), not PyPI: PyPI pygrgl lacks the "negnode matmul" fix, which
# gives wrong genetic values after generation 0. The native recombiner is
# built against the same source because the two share GRG objects.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SRC=$(dirname "$REPO")
BASE=${BASE:-$(dirname "$SRC")}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}

for d in "$SRC/grgl" "$SRC/stdpopsim_cache"; do
    [ -d "$d" ] || { echo "missing $d (unpack the whole bundle)"; exit 1; }
done

echo "== [1/5] Miniforge in $BASE/miniforge"
if [ ! -x "$BASE/miniforge/bin/conda" ]; then
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    curl -fsSL -o "$BASE/miniforge.sh" "$url" || wget -q -O "$BASE/miniforge.sh" "$url"
    bash "$BASE/miniforge.sh" -b -p "$BASE/miniforge"
    rm -f "$BASE/miniforge.sh"
fi
source "$BASE/miniforge/etc/profile.d/conda.sh"

echo "== [2/5] conda env $BASE/env (python 3.10 + C++ toolchain)"
if [ ! -x "$BASE/env/bin/python" ]; then
    conda create -y -p "$BASE/env" -c conda-forge python=3.10 cxx-compiler make cmake zlib
fi
conda activate "$BASE/env"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

echo "== [3/5] Python dependencies"
pip install -q --prefer-binary "numpy<2" scipy pandas "numba>=0.58" xarray typer rich pyyaml psutil pytest \
    nptyping funcy networkx pandas-plink dask msprime==1.2.0 tskit==0.6.4 stdpopsim==0.3.0

echo "== [4/5] pygrgl, native recombiner, xftsim (compiles C++; a few minutes)"
pip install -q "$SRC/grgl"
GRGL_ROOT="$SRC/grgl" pip install -q "$REPO/xftsim/native"
pip install -q --no-deps -e "$REPO"

cat > "$BASE/activate.sh" <<EOF
# Source this to use the xftsim scaling environment.
source "$BASE/miniforge/etc/profile.d/conda.sh"
conda activate "$BASE/env"
export STDPOPSIM_CACHE="$SRC/stdpopsim_cache"
EOF
source "$BASE/activate.sh"

echo "== [5/5] checks: imports, GRG tests, GRG vs dense on the real genetic map"
python -c "import pygrgl, grg_recomb_native, xftsim, importlib.metadata as m; print('pygrgl', m.version('pygrgl'), '| xftsim', m.version('xftsim'))"
cd "$REPO"
python -m pytest -q -p no:cacheprovider tests/pipeline/test_grg_stdpopsim_sim.py \
    tests/unit/test_grg_oracle.py tests/unit/test_grg_duplicate_mutations.py \
    tests/unit/test_crossover_sampler.py 2>&1 | tail -1
check=$(python scripts/demo_stdpopsim_grg.py --pops YRI=300,CEU=300 --length-mb 5 --gens 4 \
            --check-dense --genetic-map HapMapII_GRCh38 2>&1 | grep -E "EXACT MATCH|MISMATCH" || true)
echo "$check"
if [[ "$check" == *"EXACT MATCH"* ]]; then
    echo "READY. Submit with:  bash $HERE/submit_sweep.sh"
else
    echo "NOT READY: GRG and dense simulations disagree or the check failed; do not submit."
    exit 1
fi
