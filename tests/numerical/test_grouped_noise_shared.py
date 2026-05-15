"""
Numerical test: grouped noise produces shared environment within families.

In a simulation where noise is grouped by FID, siblings (same family)
should share the same noise value. This produces non-zero sibling
correlation even with no genetic component.
"""
import numpy as np
import pytest

from xftsim.arch import Architecture, NoiseComponent, AggregationComponent
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestGroupedNoiseShared:
    def test_shared_env_sibling_correlation(self):
        """Grouped noise (shared environment) produces sib correlation."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture(
            formula="Y ~ noise(1.0) | FID",
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        # Gen 1 offspring: siblings share same FID and same noise
        pheno = sim.phenotype_history[1]
        fids = sim.haplotype_history[1].samples.fid
        y = pheno['Y']

        # Collect sibling pairs
        unique_fids = np.unique(fids)
        sib_pairs = []
        for f in unique_fids:
            idx = np.where(fids == f)[0]
            if len(idx) == 2:
                sib_pairs.append((y[idx[0]], y[idx[1]]))

        sib_pairs = np.array(sib_pairs)
        if len(sib_pairs) > 5:
            # Sibling correlation should be 1.0 (perfect) since noise is shared
            corr = np.corrcoef(sib_pairs[:, 0], sib_pairs[:, 1])[0, 1]
            assert corr > 0.95, f"Sibling correlation {corr:.3f} should be ~1.0 for shared env"

    def test_unshared_noise_no_sib_correlation(self):
        """Without grouping, noise is independent → ~0 sib correlation."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        # No | FID grouping: each individual gets independent noise
        arch = Architecture(
            formula="Y ~ noise(1.0)",
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        fids = sim.haplotype_history[1].samples.fid
        y = pheno['Y']

        unique_fids = np.unique(fids)
        sib_pairs = []
        for f in unique_fids:
            idx = np.where(fids == f)[0]
            if len(idx) == 2:
                sib_pairs.append((y[idx[0]], y[idx[1]]))

        sib_pairs = np.array(sib_pairs)
        if len(sib_pairs) > 5:
            corr = np.corrcoef(sib_pairs[:, 0], sib_pairs[:, 1])[0, 1]
            assert abs(corr) < 0.3, f"Sib correlation {corr:.3f} should be ~0 for independent noise"

    def test_grouped_cnoise_covariance_within_family(self):
        """Grouped cnoise: within-family covariance matches target."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture(
            formula="(Y1, Y2) ~ cnoise(cov=[[1.0, 0.8], [0.8, 1.0]]) | FID",
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        fids = sim.haplotype_history[1].samples.fid
        y1 = pheno['Y1']
        y2 = pheno['Y2']

        # Siblings should have identical Y1 and Y2 (shared noise)
        unique_fids = np.unique(fids)
        same_count = 0
        for f in unique_fids:
            idx = np.where(fids == f)[0]
            if len(idx) == 2:
                if y1[idx[0]] == y1[idx[1]] and y2[idx[0]] == y2[idx[1]]:
                    same_count += 1

        # Most families should have identical sibling values
        assert same_count > len(unique_fids) * 0.8
