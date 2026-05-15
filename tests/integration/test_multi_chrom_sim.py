"""
Integration test: multi-chromosome simulation.

Tests:
1. Simulation with multi-chromosome RecombinationMap completes
2. Per-chromosome allele frequencies approximately conserved
3. Offspring genotypes are binary
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation


def _make_multi_chrom_sim(n=200, m_per_chrom=25, n_chrom=2, seed=42):
    m = m_per_chrom * n_chrom
    rng = np.random.RandomState(seed)
    sm = SampleMeta(
        iid=np.arange(n), fid=np.arange(n) // 2,
        sex=np.tile([0, 1], n // 2),
    )
    chrom = np.repeat(np.arange(n_chrom), m_per_chrom)
    vm = VariantMeta(
        vid=np.array([f'chr{c}_v{i}' for c, i in zip(chrom, range(m))]),
        chrom=chrom,
    )
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

    rmap = RecombinationMap(p=0.1, m=m, chrom=chrom)
    mate = RandomMating(offspring_per_pair=2)

    return Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap, seed=seed,
    )


class TestMultiChromSim:
    def test_simulation_completes(self):
        sim = _make_multi_chrom_sim()
        sim.run(3)
        assert sim.generation == 2
        assert np.all(np.isfinite(sim.phenotypes['Y']))

    def test_allele_freq_conserved(self):
        sim = _make_multi_chrom_sim(n=400, seed=42)
        founder_af = sim.haplotypes.genotypes.mean(axis=(0, 2))
        sim.run(3)
        offspring_af = sim.haplotypes.genotypes.mean(axis=(0, 2))
        max_diff = np.max(np.abs(founder_af - offspring_af))
        # Random mating + finite population → some drift, but should be moderate
        assert max_diff < 0.25, f"Max AF diff = {max_diff:.3f}"

    def test_genotypes_binary(self):
        sim = _make_multi_chrom_sim()
        sim.run(3)
        geno = sim.haplotypes.genotypes
        assert np.all((geno == 0) | (geno == 1))

    def test_phenotype_variance_reasonable(self):
        sim = _make_multi_chrom_sim(n=400)
        sim.run(3)
        var_y = np.var(sim.phenotypes['Y'])
        # Should have some variance
        assert var_y > 0.01
        # But not blow up
        assert var_y < 100
