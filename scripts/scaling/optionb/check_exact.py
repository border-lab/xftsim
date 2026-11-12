"""Genetic values G = X beta per generation must match the baseline run exactly (same seed)."""
import sys
import numpy as np

base = np.load(sys.argv[1])
for path in sys.argv[2:]:
    other = np.load(path)
    worst = 0.0
    for k in base.files:
        if k not in other.files:
            continue
        a, b = base[k], other[k]
        worst = max(worst, float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300)))
    gens = sorted(set(base.files) & set(other.files), key=int)
    print(f"{path}: generations 0..{gens[-1]} compared, max relative |dG| = {worst:.2e}")
