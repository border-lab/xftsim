"""
Regression tests: recurrent mutations that share (position, ref, alt).

GRG-native meiosis ends with ``sort_mutations()``, which renumbers mutations
by (position, alleles). Two mutations with identical keys tie and can come
back swapped, silently exchanging their genotype columns relative to
VariantMeta and the effect vectors. Meiosis must refuse such GRGs, and the
founder helpers must remove the sites that produce them.
"""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
msprime = pytest.importorskip("msprime")

from xftsim.founders import (
    _drop_multi_mutation_sites, _tree_sequence_to_mutable_grg,
    founder_haplotypes_from_msprime_grg,
)
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.struct import GraphHaplotypeOperator, SampleMeta


DUP_POSITION = 5000.0


def _ts_with_recurrent_mutation():
    """Small tree sequence plus one site carrying two identical A->T mutations."""
    ts = msprime.sim_ancestry(samples=6, sequence_length=10_000, population_size=1e4,
                              recombination_rate=1e-8, random_seed=3)
    ts = msprime.sim_mutations(ts, rate=2e-7, random_seed=4)
    # Start duplicate-free so the planted pair is the only one.
    ts = _drop_multi_mutation_sites(ts)
    tables = ts.dump_tables()
    assert DUP_POSITION not in set(tables.sites.position)
    site = tables.sites.add_row(position=DUP_POSITION, ancestral_state="A")
    tables.mutations.add_row(site=site, node=0, derived_state="T")
    tables.mutations.add_row(site=site, node=5, derived_state="T")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def _operator(ts):
    grg = _tree_sequence_to_mutable_grg(ts)
    return GraphHaplotypeOperator(
        grg, samples=SampleMeta(iid=np.arange(grg.num_individuals)))


def _one_meiosis(op):
    asg = RandomMating(offspring_per_pair=2).mate(op.samples, rng=np.random.RandomState(0))
    rmap = RecombinationMap(p=0.1, m=op.m, pos_bp=op.variants.pos_bp)
    return op.meiosis(asg, rmap, rng=np.random.RandomState(1))


class TestDuplicateMutationGuard:
    def test_meiosis_rejects_identical_duplicates(self):
        op = _operator(_ts_with_recurrent_mutation())
        keys = [(op._grg.get_mutation_by_id(i).position,
                 op._grg.get_mutation_by_id(i).ref_allele,
                 op._grg.get_mutation_by_id(i).allele) for i in range(op.m)]
        # The planted pair is the only duplicate, so the guard must catch it.
        assert len(keys) - len(set(keys)) == 1
        assert keys.count((int(DUP_POSITION), "A", "T")) == 2
        with pytest.raises(ValueError, match="1 mutation"):
            _one_meiosis(op)

    def test_dropping_multi_mutation_sites_allows_meiosis(self):
        ts = _drop_multi_mutation_sites(_ts_with_recurrent_mutation())
        assert all(len(site.mutations) == 1 for site in ts.sites())
        assert DUP_POSITION not in {site.position for site in ts.sites()}
        offspring = _one_meiosis(_operator(ts))
        assert isinstance(offspring, GraphHaplotypeOperator)

    def test_drop_is_noop_without_multi_mutation_sites(self):
        ts = msprime.sim_ancestry(samples=4, sequence_length=1_000, population_size=1e4,
                                  random_seed=1)
        ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=2,
                                   discrete_genome=False)
        assert _drop_multi_mutation_sites(ts) is ts


class TestMsprimeFounders:
    def test_default_founders_have_one_mutation_per_position(self):
        op = founder_haplotypes_from_msprime_grg(
            n=50, sequence_length=200_000, mutation_rate=1e-7, seed=11)
        assert np.all(np.diff(op.variants.pos_bp) > 0)

    def test_seed_reproducible(self):
        a = founder_haplotypes_from_msprime_grg(n=10, sequence_length=50_000,
                                                mutation_rate=1e-7, seed=5)
        b = founder_haplotypes_from_msprime_grg(n=10, sequence_length=50_000,
                                                mutation_rate=1e-7, seed=5)
        np.testing.assert_array_equal(a.variants.pos_bp, b.variants.pos_bp)
        np.testing.assert_array_equal(a.to_dense().genotypes, b.to_dense().genotypes)
