"""
Metadata inheritance on GRG-backed stdpopsim founders.

The GRG and dense backends must produce identical per-individual metadata
every generation, and offspring row i of a GRG generation must be the child
of the parents recorded for row i (which is what metadata inheritance relies
on): wherever a recorded parent is homozygous, the offspring's haplotype
from that parent must carry that allele, whatever the recombination.
"""
import numpy as np
import pytest

pytest.importorskip("pygrgl")
pytest.importorskip("msprime")
pytest.importorskip("stdpopsim")

from xftsim.arch import Architecture
from xftsim.effect import AdditiveEffects
from xftsim.founders import founder_haplotypes_from_stdpopsim_grg
from xftsim.inherit import AncestryProportions, Maternal
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import MetadataSummary
from xftsim.struct import DenseHaplotypeArray, GraphHaplotypeOperator

N_GEN = 4
SEED = 3
RULES = [AncestryProportions('population'), Maternal('population')]


@pytest.fixture(scope="module")
def founders():
    return founder_haplotypes_from_stdpopsim_grg(
        samples={"YRI": 60, "CEU": 60}, chromosome="chr22",
        left=20_000_000, right=21_000_000, seed=SEED,
    )


def _sim(founder, rmap):
    eff = AdditiveEffects.from_h2(h2=0.5, m=founder.m, seed=SEED)
    arch = Architecture.from_formula("""
    Y.G ~ genetic(eff)
    Y.A ~ meta(ancestry_YRI)
    Y ~ Y.G + Y.A
    """, effects={'eff': eff})
    return Simulation(founder_haplotypes=founder, architecture=arch,
                      mating_regime=RandomMating(offspring_per_pair=2),
                      recombination_map=rmap, seed=SEED, retain_haplotypes=1,
                      retain_phenotypes=N_GEN, metadata_rules=RULES,
                      statistics=[MetadataSummary(by='population')])


@pytest.fixture(scope="module")
def sims(founders):
    rmap = RecombinationMap.from_haplotypes_with_cM(founders)
    dense = founders.to_dense()
    dense_founders = DenseHaplotypeArray(genotypes=dense.genotypes.copy(),
                                         samples=founders.samples,
                                         variants=founders.variants)
    sim_dense = _sim(dense_founders, rmap)
    sim_dense.run(N_GEN)
    sim_grg = _sim(founders, rmap)
    sim_grg.run(N_GEN)
    return sim_grg, sim_dense


class TestStdpopsimGRGMetadata:
    def test_founders_carry_population(self, founders):
        assert set(np.unique(founders.samples.extra['population'])) == {'YRI', 'CEU'}

    def test_grg_and_dense_metadata_identical(self, sims):
        sim_grg, sim_dense = sims
        assert isinstance(sim_grg.haplotypes, GraphHaplotypeOperator)
        for gen in range(N_GEN):
            g = sim_grg.phenotype_history[gen].samples.extra
            d = sim_dense.phenotype_history[gen].samples.extra
            assert set(g) == {'population', 'ancestry_YRI', 'ancestry_CEU'}
            assert set(g) == set(d)
            for k in g:
                np.testing.assert_array_equal(g[k], d[k])
            np.testing.assert_array_equal(g['ancestry_YRI'] + g['ancestry_CEU'], 1.0)
        assert ([r.statistics['MetadataSummary'] for r in sim_grg.results]
                == [r.statistics['MetadataSummary'] for r in sim_dense.results])

    def test_grg_offspring_rows_match_recorded_parents(self, sims):
        sim_grg, sim_dense = sims
        gen = N_GEN - 1
        # GRG meiosis mutates the shared graph, so the GRG parent operator is
        # stale; the dense run has the same parent genotypes.
        parents = sim_dense.haplotype_history[gen - 1].genotypes
        child = sim_grg.haplotype_history[gen].to_dense().genotypes
        ped = sim_grg.pedigree_history[gen]
        for hap, idx in ((0, ped.maternal_idx), (1, ped.paternal_idx)):
            par = parents[idx]
            hom = par[:, :, 0] == par[:, :, 1]
            assert hom.mean() > 0.5
            np.testing.assert_array_equal(child[:, :, hap][hom], par[:, :, 0][hom])
