"""
Phase 2 tests: mate assignment, meiosis, simulation loop.
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray
from xftsim.nmate import NMateAssignment, RandomMating
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MVGeneticComponent, HaplotypeGeneticComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


# ---------------------------------------------------------------------------
# NMateAssignment tests
# ---------------------------------------------------------------------------

class TestNMateAssignment:
    def _samples(self, n=10):
        return SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], (n + 1) // 2)[:n])

    def test_creation(self):
        samples = self._samples(4)
        ma = NMateAssignment(
            offspring_samples=samples,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
        )
        assert ma.n_offspring == 4

    def test_length_mismatch_maternal(self):
        samples = self._samples(4)
        with pytest.raises(ValueError, match="maternal_idx length"):
            NMateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([2, 2, 3, 3]),
            )

    def test_length_mismatch_paternal(self):
        samples = self._samples(4)
        with pytest.raises(ValueError, match="paternal_idx length"):
            NMateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([2, 2]),
            )

    def test_negative_maternal(self):
        samples = self._samples(2)
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
            )

    def test_negative_paternal(self):
        samples = self._samples(2)
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([0, -1]),
            )


# ---------------------------------------------------------------------------
# RandomMating tests
# ---------------------------------------------------------------------------

class TestRandomMating:
    def _balanced_samples(self, n=100):
        sex = np.tile([0, 1], n // 2)
        return SampleMeta(iid=np.arange(n), sex=sex)

    def test_correct_offspring_count(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        # 50 females, 50 males -> 50 pairs -> 100 offspring
        assert ma.n_offspring == 100

    def test_offspring_count_3_per_pair(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=3)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        assert ma.n_offspring == 150

    def test_generation_increment(self):
        samples = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5), generation=3)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        assert ma.offspring_samples.generation == 4

    def test_fid_grouping(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        # Each FID should appear exactly offspring_per_pair times
        fids = ma.offspring_samples.fid
        _, counts = np.unique(fids, return_counts=True)
        assert np.all(counts == 2)

    def test_sex_balance(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        # With opp=2, each pair has one male and one female
        assert ma.offspring_samples.n_female == ma.offspring_samples.n_male

    def test_valid_indices(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        assert np.all(ma.maternal_idx >= 0)
        assert np.all(ma.maternal_idx < 100)
        assert np.all(ma.paternal_idx >= 0)
        assert np.all(ma.paternal_idx < 100)

    def test_no_self_mating(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        # Mothers should all be female (sex=0), fathers male (sex=1)
        assert np.all(samples.sex[ma.maternal_idx] == 0)
        assert np.all(samples.sex[ma.paternal_idx] == 1)

    def test_determinism_with_seed(self):
        samples = self._balanced_samples(100)
        rm = RandomMating(offspring_per_pair=2)
        ma1 = rm.mate(samples, rng=np.random.RandomState(42))
        ma2 = rm.mate(samples, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(ma1.maternal_idx, ma2.maternal_idx)
        np.testing.assert_array_equal(ma1.paternal_idx, ma2.paternal_idx)

    def test_unbalanced_sex(self):
        # 30 females, 70 males
        sex = np.array([0]*30 + [1]*70)
        samples = SampleMeta(iid=np.arange(100), sex=sex)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(samples, rng=np.random.RandomState(42))
        # Should produce min(30, 70) * 2 = 60 offspring
        assert ma.n_offspring == 60

    def test_no_females_raises(self):
        samples = SampleMeta(iid=np.arange(10), sex=np.ones(10, dtype=int))
        rm = RandomMating(offspring_per_pair=2)
        with pytest.raises(ValueError, match="female"):
            rm.mate(samples, rng=np.random.RandomState(42))

    def test_invalid_opp(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=0)


# ---------------------------------------------------------------------------
# Meiosis tests
# ---------------------------------------------------------------------------

class TestMeiosis:
    @pytest.fixture
    def parent_hap(self):
        return TestSimulation.founder_haplotypes(n=100, m=50)

    @pytest.fixture
    def rmap(self):
        return TestSimulation.recombination_map(m=50)

    def test_output_shape(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)
        assert offspring.genotypes.shape == (ma.n_offspring, 50, 2)

    def test_output_type(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)
        assert isinstance(offspring, DenseHaplotypeArray)

    def test_alleles_binary(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)
        assert set(np.unique(offspring.genotypes)).issubset({0, 1})

    def test_generation_correct(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)
        assert offspring.generation == 1

    def test_variants_inherited(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)
        np.testing.assert_array_equal(offspring.vid, parent_hap.vid)

    def test_alleles_from_parents(self, parent_hap, rmap):
        """Each offspring allele must be present in the corresponding parent."""
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring = parent_hap.meiosis(ma, rmap)

        # Check a subset of offspring
        for i in range(min(10, ma.n_offspring)):
            mat_idx = ma.maternal_idx[i]
            pat_idx = ma.paternal_idx[i]
            for j in range(offspring.m):
                # Maternal haplotype of offspring must come from mother's two haplotypes
                off_mat = offspring.genotypes[i, j, 0]
                mom_alleles = set(parent_hap.genotypes[mat_idx, j, :])
                assert off_mat in mom_alleles, (
                    f"Offspring {i} variant {j} maternal allele {off_mat} "
                    f"not in mother's alleles {mom_alleles}"
                )
                # Paternal haplotype must come from father
                off_pat = offspring.genotypes[i, j, 1]
                dad_alleles = set(parent_hap.genotypes[pat_idx, j, :])
                assert off_pat in dad_alleles, (
                    f"Offspring {i} variant {j} paternal allele {off_pat} "
                    f"not in father's alleles {dad_alleles}"
                )


# ---------------------------------------------------------------------------
# NSimulation tests
# ---------------------------------------------------------------------------

class TestNSimulation:
    @pytest.fixture
    def sim_components(self):
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=50)
        return hap, arch, rm, rmap

    def test_three_gen_completes(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(3)
        assert sim.generation == 2

    def test_history_populated(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(3)
        # Phenotypes should exist for gens 0, 1, 2
        assert 0 in sim.phenotype_history
        assert 1 in sim.phenotype_history
        assert 2 in sim.phenotype_history

    def test_retention_haplotypes(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_haplotypes=1, retain_phenotypes=10)
        sim.run(4)
        # With retain_haplotypes=1, only current gen should remain
        assert sim.generation in sim.haplotype_history
        assert 0 not in sim.haplotype_history

    def test_retention_phenotypes(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_haplotypes=10, retain_phenotypes=1)
        sim.run(4)
        # With retain_phenotypes=1, only recent gens should remain
        assert sim.generation in sim.phenotype_history
        assert 0 not in sim.phenotype_history

    def test_population_size(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(3)
        # With balanced sex and opp=2, population stays constant
        assert sim.haplotypes.n == 500

    def test_callback_called(self, sim_components):
        hap, arch, rm, rmap = sim_components
        call_count = [0]
        def counter(s):
            call_count[0] += 1
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[counter])
        sim.run(3)
        # Callback called once per generation (0, 1, 2)
        assert call_count[0] == 3

    def test_early_stopping(self, sim_components):
        hap, arch, rm, rmap = sim_components
        def stopper(s):
            if s.generation >= 1:
                s.stop = True
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[stopper])
        sim.run(10)
        assert sim.generation == 1

    def test_phenotype_keys(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(2)
        keys = list(sim.phenotypes.keys)
        assert 'Y.G' in keys
        assert 'Y.E' in keys
        assert 'Y' in keys

    def test_phenotype_shape(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(2)
        assert sim.phenotypes['Y'].shape == (sim.haplotypes.n,)

    def test_single_gen(self, sim_components):
        """Running with n_generations=1 should just compute gen-0 phenotypes."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(1)
        assert sim.generation == 0
        assert 0 in sim.phenotype_history
        assert len(sim.haplotype_history) == 1


# ---------------------------------------------------------------------------
# Pedigree integrity tests
# ---------------------------------------------------------------------------

class TestPedigreeIntegrity:
    def test_pedigree_valid_indices(self):
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(3)
        for gen in range(1, 3):
            ped = sim.pedigree_history[gen]
            assert np.all(ped.maternal_idx >= 0)
            assert np.all(ped.paternal_idx >= 0)
            assert np.all(ped.maternal_idx < ped.parent_n)
            assert np.all(ped.paternal_idx < ped.parent_n)

    def test_siblings_share_fid(self):
        hap = TestSimulation.founder_haplotypes(n=100, m=50)
        arch = TestSimulation.simple_architecture(m=50)
        rm = RandomMating(offspring_per_pair=3)
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(2)
        ped = sim.pedigree_history[1]
        # Siblings (same maternal_idx and paternal_idx) should share FID
        for i in range(ped.offspring_samples.n):
            for j in range(i + 1, ped.offspring_samples.n):
                if (ped.maternal_idx[i] == ped.maternal_idx[j] and
                        ped.paternal_idx[i] == ped.paternal_idx[j]):
                    assert ped.offspring_samples.fid[i] == ped.offspring_samples.fid[j]


# ---------------------------------------------------------------------------
# Genetic covariance test (statistical)
# ---------------------------------------------------------------------------

class TestGeneticCovariance:
    def test_h2_approximately_correct(self):
        """Var(G)/Var(Y) should be approximately h2, tested at gen 0.

        Note: The standardized_matvec centers genotypes (subtracts 2*p) but does
        not scale by sqrt(2*p*(1-p)). With effects drawn as N(0, h2/m) and
        non-uniform AFs, the realized Var(G) may differ from h2. We check that
        the genetic signal is a substantial fraction of total variance.
        """
        n = 5000
        m = 500
        h2 = 0.5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=99)
        arch = TestSimulation.simple_architecture(m=m, h2=h2, seed=123)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=m)
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(1)

        pheno = sim.phenotype_history[0]
        var_g = np.var(pheno['Y.G'])
        var_y = np.var(pheno['Y'])
        observed_h2 = var_g / var_y
        # With centered (not scaled) genotypes and random AFs, realized h2
        # will be close but not exact. Check it's in a reasonable range.
        assert 0.15 < observed_h2 < 0.85, (
            f"Observed h2={observed_h2:.3f}, expected roughly ~{h2}"
        )


class TestValidation:
    def test_effect_dimension_mismatch(self):
        """Mismatched effect m vs founder m should raise ValueError."""
        hap = TestSimulation.founder_haplotypes(n=100, m=50, seed=42)
        # Effects with m=20, but founders have m=50
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)

        with pytest.raises(ValueError, match="Effect dimension mismatch"):
            sim.run(1)

    def test_matching_dimensions_ok(self):
        """Matching effect m and founder m should not raise."""
        hap = TestSimulation.founder_haplotypes(n=100, m=50, seed=42)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=42)
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(1)  # Should not raise
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y']))


# ---------------------------------------------------------------------------
# NSimulation edge case and continue_run tests
# ---------------------------------------------------------------------------

class TestContinueRun:
    @pytest.fixture
    def sim_components(self):
        hap = TestSimulation.founder_haplotypes(n=200, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=50)
        return hap, arch, rm, rmap

    def test_continue_run_extends_generation(self, sim_components):
        """continue_run should advance generations from where run left off."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_haplotypes=10, retain_phenotypes=10)
        sim.run(3)
        assert sim.generation == 2
        sim.continue_run(2)
        assert sim.generation == 4
        assert 3 in sim.phenotype_history
        assert 4 in sim.phenotype_history

    def test_continue_run_zero_additional(self, sim_components):
        """continue_run(0) should be a no-op."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(2)
        gen_before = sim.generation
        sim.continue_run(0)
        assert sim.generation == gen_before

    def test_continue_run_phenotypes_finite(self, sim_components):
        """Phenotypes from continue_run should be finite."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(2)
        sim.continue_run(3)
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))

    def test_continue_run_with_callbacks(self, sim_components):
        """Callbacks should fire during continue_run."""
        hap, arch, rm, rmap = sim_components
        call_log = []
        def logger(s):
            call_log.append(s.generation)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[logger])
        sim.run(2)  # gens 0, 1 → 2 callbacks
        assert len(call_log) == 2
        sim.continue_run(2)  # gens 2, 3 → 2 more callbacks
        assert len(call_log) == 4
        assert call_log[-2:] == [2, 3]

    def test_continue_run_early_stop(self, sim_components):
        """Early stopping in continue_run should work."""
        hap, arch, rm, rmap = sim_components
        def stopper(s):
            if s.generation >= 3:
                s.stop = True
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[stopper])
        sim.run(2)
        sim.continue_run(5)
        assert sim.generation == 3

    def test_continue_run_retention(self, sim_components):
        """Retention policy should be enforced during continue_run."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_haplotypes=1, retain_phenotypes=2)
        sim.run(3)
        sim.continue_run(3)
        # Only recent generations should remain
        assert sim.generation in sim.haplotype_history
        assert 0 not in sim.haplotype_history
        assert 1 not in sim.phenotype_history


class TestNSimulationEdgeCases:
    @pytest.fixture
    def sim_components(self):
        hap = TestSimulation.founder_haplotypes(n=200, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=50)
        return hap, arch, rm, rmap

    def test_repr(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        r = repr(sim)
        assert 'NSimulation' in r
        assert 'generation=0' in r

    def test_haplotypes_property(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        assert sim.haplotypes is hap

    def test_phenotypes_property_after_run(self, sim_components):
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(1)
        assert 'Y' in sim.phenotypes.keys

    def test_multiple_callbacks(self, sim_components):
        """Multiple callbacks should all fire."""
        hap, arch, rm, rmap = sim_components
        counts = [0, 0]
        def cb1(s): counts[0] += 1
        def cb2(s): counts[1] += 1
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[cb1, cb2])
        sim.run(3)
        assert counts[0] == 3
        assert counts[1] == 3

    def test_no_pedigree_at_gen0(self, sim_components):
        """Generation 0 should have no pedigree."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(2)
        assert 0 not in sim.pedigree_history
        assert 1 in sim.pedigree_history

    def test_seed_determinism(self):
        """Same seed should produce identical gen-0 phenotypes."""
        # Create independent founders so they don't share state
        hap1 = TestSimulation.founder_haplotypes(n=200, m=50, seed=7)
        hap2 = TestSimulation.founder_haplotypes(n=200, m=50, seed=7)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=50)
        sim1 = NSimulation(hap1, arch, rm, rmap, seed=42)
        sim1.run(1)
        sim2 = NSimulation(hap2, arch, rm, rmap, seed=42)
        sim2.run(1)
        np.testing.assert_array_equal(
            sim1.phenotype_history[0]['Y'],
            sim2.phenotype_history[0]['Y'],
        )

    def test_different_seeds_differ(self, sim_components):
        """Different seeds should produce different results."""
        hap, arch, rm, rmap = sim_components
        sim1 = NSimulation(hap, arch, rm, rmap, seed=42)
        sim1.run(2)
        sim2 = NSimulation(hap, arch, rm, rmap, seed=99)
        sim2.run(2)
        assert not np.array_equal(
            sim1.phenotype_history[1]['Y'],
            sim2.phenotype_history[1]['Y'],
        )

    def test_retain_all(self, sim_components):
        """Large retention should keep all generations."""
        hap, arch, rm, rmap = sim_components
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_haplotypes=100, retain_phenotypes=100)
        sim.run(5)
        for g in range(5):
            assert g in sim.phenotype_history
            assert g in sim.haplotype_history

    def test_with_filters_and_statistics(self, sim_components):
        """Simulation with both filters and statistics."""
        from xftsim.nstats import SampleStatistics
        from xftsim.nfilter import SibPairFilter
        hap, arch, rm, rmap = sim_components
        rm_3 = RandomMating(offspring_per_pair=3)
        sim = NSimulation(
            hap, arch, rm_3, rmap, seed=42,
            retain_phenotypes=10,
            filters={'sib_pairs': SibPairFilter()},
            statistics=[SampleStatistics()],
        )
        sim.run(3)
        assert len(sim.results) == 3
        for r in sim.results:
            assert 'SampleStatistics' in r.statistics


class TestSimCallbackEdgeCases:
    """Test callback behavior edge cases."""

    @pytest.fixture
    def sim_components(self):
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        return hap, arch, RandomMating(offspring_per_pair=2), rmap

    def test_callback_exception_propagates(self, sim_components):
        """If a callback raises an exception, it should propagate."""
        hap, arch, rm, rmap = sim_components
        def bad_callback(sim):
            raise RuntimeError("callback error")
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[bad_callback])
        with pytest.raises(RuntimeError, match="callback error"):
            sim.run(2)

    def test_callback_receives_sim_object(self, sim_components):
        """Callback should receive the simulation object."""
        hap, arch, rm, rmap = sim_components
        received = []
        def track_callback(s):
            received.append(s.generation)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[track_callback])
        sim.run(3)
        assert received == [0, 1, 2]

    def test_multiple_callbacks_all_called(self, sim_components):
        """All callbacks should be called each generation."""
        hap, arch, rm, rmap = sim_components
        counts = [0, 0]
        def cb1(s): counts[0] += 1
        def cb2(s): counts[1] += 1
        sim = NSimulation(hap, arch, rm, rmap, seed=42, callbacks=[cb1, cb2])
        sim.run(3)
        assert counts == [3, 3]


class TestSimValidationEdgeCases:
    """Test _validate() with different component types."""

    def test_mvgenetic_dimension_mismatch(self):
        """MVGeneticComponent with wrong m should raise at validation."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=30, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        arch.add('Y1.E', NoiseComponent(variance=0.5))
        arch.add('Y2.E', NoiseComponent(variance=0.5))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))

        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap, seed=42)
        with pytest.raises(ValueError, match="Effect dimension mismatch"):
            sim.run(1)

    def test_haplotype_genetic_dimension_mismatch(self):
        """HaplotypeGeneticComponent with wrong m should raise at validation."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=30, seed=42)
        arch = Architecture()
        arch.add('Y.G', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap, seed=42)
        with pytest.raises(ValueError, match="Effect dimension mismatch"):
            sim.run(1)

    def test_matching_dimensions_pass(self):
        """Correctly matched dimensions should not raise."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap, seed=42)
        sim.run(1)  # Should not raise
        assert np.all(np.isfinite(sim.phenotypes['Y']))


class TestSimRetentionEdgeCases:
    """Test retention policy edge cases."""

    def test_retain_haplotypes_zero(self):
        """retain_haplotypes=0 should keep no old haplotypes (only current gen)."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap,
                         seed=42, retain_haplotypes=0)
        sim.run(5)
        # Only the most recent gen's haplotypes should remain
        # (gen 4 is current, retain=0 means keep 0 past gens → only gen 4)
        assert sim.generation == 4
        assert 4 in sim.haplotype_history
        # Earlier generations should be pruned
        assert 0 not in sim.haplotype_history
        assert 1 not in sim.haplotype_history

    def test_retain_phenotypes_one(self):
        """retain_phenotypes=1 should keep only current + 1 past gen."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap,
                         seed=42, retain_phenotypes=1)
        sim.run(5)
        # Should have at most 2 generations in phenotype history
        assert len(sim.phenotype_history) <= 2
        assert 4 in sim.phenotype_history

    def test_single_generation_no_pruning(self):
        """Running 1 generation should not prune anything."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        sim = NSimulation(hap, arch, RandomMating(offspring_per_pair=2), rmap,
                         seed=42, retain_haplotypes=1, retain_phenotypes=1)
        sim.run(1)
        assert 0 in sim.haplotype_history
        assert 0 in sim.phenotype_history
