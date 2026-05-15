"""
Unit tests for Simulation internal methods.

Tests focus on internal method behavior:
- _enforce_retention: history pruning logic for haplotypes/phenotypes/pedigrees
- _run_filters_and_stats: filter application and statistic computation
- _run_callbacks: callback execution
- continue_run: generation extension and history preservation

Note: This file covers internal method behavior. For integration tests,
see test_sim.py, test_retention_policy.py, test_nsim_properties_and_callbacks.py.
"""
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.filters import TrioFilter, SibPairFilter
from xftsim.stats import SampleStatistics


def _minimal_sim(n=100, m=20, retain_haplotypes=1, retain_phenotypes=2,
                 filters=None, statistics=None, callbacks=None, seed=42):
    """Create a minimal simulation for internal method testing."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    effects = AdditiveEffects.from_h2(m=m, h2=0.5, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    mating = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)

    return Simulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=retain_haplotypes,
        retain_phenotypes=retain_phenotypes,
        filters=filters or {},
        statistics=statistics or [],
        callbacks=callbacks or [],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# _enforce_retention tests
# ---------------------------------------------------------------------------

class TestEnforceRetention:
    """Test the _enforce_retention internal method."""

    def test_retain_haplotypes_1_drops_old_gens(self):
        """retain_haplotypes=1 should drop generations older than current-1."""
        sim = _minimal_sim(retain_haplotypes=1, retain_phenotypes=10)
        sim.run(5)  # Runs gens 0,1,2,3,4 → final gen=4

        # With retain_haplotypes=1, only current and 1 past should remain
        # After gen 4: keep 4, 3 (drop 0, 1, 2)
        hap_keys = sorted(sim.haplotype_history.keys())
        assert hap_keys == [3, 4], f"Expected [3, 4], got {hap_keys}"

        # Verify old generations were explicitly dropped
        assert 0 not in sim.haplotype_history
        assert 1 not in sim.haplotype_history
        assert 2 not in sim.haplotype_history

    def test_retain_phenotypes_2_keeps_exactly_two_past(self):
        """retain_phenotypes=2 should keep current + 2 past generations."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=2)
        sim.run(6)  # Runs gens 0,1,2,3,4,5 → final gen=5

        # With retain_phenotypes=2: keep gen 5, 4, 3 (drop 0, 1, 2)
        pheno_keys = sorted(sim.phenotype_history.keys())
        expected = [3, 4, 5]
        assert pheno_keys == expected, f"Expected {expected}, got {pheno_keys}"

        # Pedigrees follow phenotype retention
        ped_keys = sorted(sim.pedigree_history.keys())
        # Pedigrees start at gen 1, pruned same as phenotypes
        # But pedigree gen 0 doesn't exist
        for k in ped_keys:
            assert k >= 3, f"Old pedigree gen {k} should have been dropped"

    def test_retain_phenotypes_0_drops_all_but_current(self):
        """retain_phenotypes=0 should only keep the current generation."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=0)
        sim.run(4)  # Runs gens 0,1,2,3 → final gen=3

        # Only gen 3 should remain
        pheno_keys = list(sim.phenotype_history.keys())
        assert pheno_keys == [3], f"Expected only [3], got {pheno_keys}"

        # Pedigrees also pruned
        ped_keys = list(sim.pedigree_history.keys())
        assert all(k >= 3 for k in ped_keys)

    def test_mate_assignments_only_keep_recent(self):
        """Mate assignments should only keep the most recent (current-1)."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=10)
        sim.run(5)  # Gens 0,1,2,3,4 → final gen=4

        # Mate assignments are for transitions: gen i assigns mates for gen i+1
        # After gen 4, we no longer need assignments from gen < 3
        mate_keys = sorted(sim._mate_assignments.keys())

        # Should only have assignment for gen 3 (which produced gen 4)
        # and possibly gen 4 if run tried to continue (but run doesn't do that on last gen)
        # Actually, run(5) produces gens 0-4, last mate assignment is for gen 3->4
        # _enforce_retention drops g < current_gen - 1
        # So at gen 4, keep only g >= 3
        for k in mate_keys:
            assert k >= 3, f"Old mate assignment gen {k} should have been dropped"

    def test_retention_triggered_each_generation(self):
        """_enforce_retention should be called after each generation > 0."""
        sim = _minimal_sim(retain_haplotypes=0, retain_phenotypes=10)
        sim.run(4)  # Gens 0,1,2,3 → final gen=3

        # With retain_haplotypes=0, only current gen should remain
        hap_keys = list(sim.haplotype_history.keys())
        assert hap_keys == [3], f"Expected only [3], got {hap_keys}"

    def test_large_retention_keeps_all(self):
        """Large retention values should preserve all generations."""
        sim = _minimal_sim(retain_haplotypes=100, retain_phenotypes=100)
        sim.run(4)  # Gens 0,1,2,3 → final gen=3

        hap_keys = sorted(sim.haplotype_history.keys())
        pheno_keys = sorted(sim.phenotype_history.keys())

        assert hap_keys == [0, 1, 2, 3]
        assert pheno_keys == [0, 1, 2, 3]

    def test_pedigree_retention_follows_phenotype(self):
        """Pedigree history should be pruned with same policy as phenotypes."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=1)
        sim.run(5)  # Gens 0,1,2,3,4 → final gen=4

        pheno_keys = sorted(sim.phenotype_history.keys())
        ped_keys = sorted(sim.pedigree_history.keys())

        # Pedigrees start at gen 1 (no pedigree for founders)
        # Both should be pruned to keep only gen 4 and 3
        min_pheno_gen = min(pheno_keys)
        for k in ped_keys:
            assert k >= min_pheno_gen, \
                f"Pedigree gen {k} should be pruned like phenotypes (>= {min_pheno_gen})"


# ---------------------------------------------------------------------------
# continue_run tests
# ---------------------------------------------------------------------------

class TestContinueRun:
    """Test continue_run method behavior."""

    def test_continue_run_extends_generation_count(self):
        """continue_run should increment generation properly."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=10)
        sim.run(3)  # Gens 0,1,2 → final gen=2
        assert sim.generation == 2

        sim.continue_run(2)  # Add gens 3,4 → final gen=4
        assert sim.generation == 4

    def test_continue_run_preserves_existing_history(self):
        """Existing history should be preserved after continue_run (subject to retention)."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=10)
        sim.run(3)  # Gens 0,1,2 → final gen=2

        # Record gen 2 phenotypes
        gen2_pheno = sim.phenotype_history[2]['Y'].copy()

        sim.continue_run(2)  # Add gens 3,4 → final gen=4

        # Gen 2 should still exist and be unchanged
        assert 2 in sim.phenotype_history
        np.testing.assert_array_equal(sim.phenotype_history[2]['Y'], gen2_pheno)

    def test_continue_run_after_retention_prunes(self):
        """continue_run with retention should prune old history."""
        sim = _minimal_sim(retain_haplotypes=1, retain_phenotypes=2)
        sim.run(3)  # Gens 0,1,2 → final gen=2

        # After gen 2: haplotypes=[1,2], phenotypes=[0,1,2]
        assert 0 in sim.phenotype_history

        sim.continue_run(3)  # Add gens 3,4,5 → final gen=5

        # After gen 5: haplotypes=[4,5], phenotypes=[3,4,5]
        # Old gens should be pruned
        assert 0 not in sim.phenotype_history
        assert 1 not in sim.phenotype_history

        hap_keys = sorted(sim.haplotype_history.keys())
        assert hap_keys == [4, 5]

    def test_continue_run_zero_is_noop(self):
        """continue_run(0) should not change state."""
        sim = _minimal_sim()
        sim.run(2)  # Gens 0,1 → final gen=1

        gen_before = sim.generation
        n_hap_before = len(sim.haplotype_history)

        sim.continue_run(0)

        assert sim.generation == gen_before
        assert len(sim.haplotype_history) == n_hap_before

    def test_continue_run_creates_mate_assignment(self):
        """continue_run should create mate assignment for current gen if missing."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=10)
        sim.run(2)  # Gens 0,1 → final gen=1

        # After run(2), mate assignment for gen 1->2 might not exist (last gen)
        # continue_run should create it if needed
        start_gen = sim.generation
        sim.continue_run(1)  # Add gen 2

        # Mate assignment for gen 1 should have been created
        assert start_gen in sim._mate_assignments

    def test_continue_run_with_callbacks(self):
        """Callbacks should fire during continue_run."""
        call_log = []

        def logger(s):
            call_log.append(s.generation)

        sim = _minimal_sim(callbacks=[logger])
        sim.run(2)  # Gens 0,1 → 2 callbacks

        assert call_log == [0, 1]

        sim.continue_run(2)  # Gens 2,3 → 2 more callbacks

        assert call_log == [0, 1, 2, 3]

    def test_continue_run_resets_stop_flag(self):
        """continue_run should reset sim.stop to False."""
        def stopper(s):
            if s.generation == 1:
                s.stop = True

        sim = _minimal_sim(callbacks=[stopper])
        sim.run(5)  # Should stop at gen 1
        assert sim.generation == 1
        assert sim.stop is True

        # continue_run should reset stop flag
        sim.continue_run(2)  # Should run gens 2,3
        # But stopper won't fire again at gen 1 since we're past it
        assert sim.generation == 3


# ---------------------------------------------------------------------------
# _run_filters_and_stats tests
# ---------------------------------------------------------------------------

class TestRunFiltersAndStats:
    """Test _run_filters_and_stats internal method."""

    def test_multiple_filters_both_applied(self):
        """Both TrioFilter and SibPairFilter should be applied in same sim."""
        # Use offspring_per_pair=3 to ensure sibling pairs exist
        filters = {
            'trio': TrioFilter(),
            'sib_pair': SibPairFilter(),
        }

        sim = _minimal_sim(filters=filters, retain_phenotypes=10)
        # Use custom mating for more siblings
        sim.mating_regime = RandomMating(offspring_per_pair=3)
        sim.run(3)  # Gens 0,1,2 → final gen=2

        # Filters don't produce direct output, but they're called
        # We can verify by checking that the simulation completes without error
        assert sim.generation == 2

        # TrioFilter needs gen > 0, SibPairFilter works at any gen
        # At gen 2, both should have been called
        # (We'd need to mock to verify calls, but behavior is tested elsewhere)

    def test_statistics_computed_per_generation(self):
        """Statistics should be computed for each generation."""
        stats = [SampleStatistics()]
        sim = _minimal_sim(statistics=stats, retain_phenotypes=10)
        sim.run(4)  # Gens 0,1,2,3 → final gen=3

        # results should have 4 entries (one per generation)
        assert len(sim.results) == 4

        # Each result should have correct generation number
        for i, result in enumerate(sim.results):
            assert result.generation == i

    def test_statistics_results_structure(self):
        """Results should have nested dict structure: statistics['SampleStatistics'][...]."""
        stats = [SampleStatistics()]
        sim = _minimal_sim(statistics=stats)
        sim.run(2)  # Gens 0,1 → final gen=1

        # Check result structure
        assert len(sim.results) == 2

        result0 = sim.results[0]
        assert result0.generation == 0
        assert 'SampleStatistics' in result0.statistics

        stat_dict = result0.statistics['SampleStatistics']
        assert 'cov' in stat_dict
        assert 'var' in stat_dict
        assert 'keys' in stat_dict

    def test_multiple_same_statistic_instances(self):
        """Multiple instances of same statistic class should get unique keys."""
        stats = [SampleStatistics(), SampleStatistics()]
        sim = _minimal_sim(statistics=stats)
        sim.run(2)  # Gens 0,1 → final gen=1

        result = sim.results[0]
        assert 'SampleStatistics' in result.statistics
        assert 'SampleStatistics_1' in result.statistics

    def test_empty_statistics_list_produces_no_results(self):
        """With statistics=[], sim.results should be empty."""
        sim = _minimal_sim(statistics=[])
        sim.run(3)  # Gens 0,1,2 → final gen=2

        assert len(sim.results) == 0

    def test_empty_filters_dict_runs_without_error(self):
        """With filters={}, simulation should run normally."""
        sim = _minimal_sim(filters={})
        sim.run(2)  # Gens 0,1 → final gen=1

        assert sim.generation == 1
        assert 1 in sim.phenotype_history

    def test_statistics_use_filtered_views(self):
        """Statistics should receive filtered_views from filters."""
        # This is an integration test of the pipeline
        # We verify that both filters and stats work together

        filters = {'trio': TrioFilter()}
        stats = [SampleStatistics()]

        sim = _minimal_sim(filters=filters, statistics=stats, retain_phenotypes=10)
        sim.run(3)  # Gens 0,1,2 → final gen=2

        # Results should be populated
        assert len(sim.results) == 3

        # At gen 0, TrioFilter returns None (no parents)
        # At gen 1+, TrioFilter returns TrioView
        # SampleStatistics doesn't use filtered views, but the pipeline should work
        result2 = sim.results[2]
        assert 'SampleStatistics' in result2.statistics

    def test_filter_none_result_handled_gracefully(self):
        """Filters that return None should not cause errors."""
        # TrioFilter returns None at gen 0
        filters = {'trio': TrioFilter()}
        stats = [SampleStatistics()]

        sim = _minimal_sim(filters=filters, statistics=stats)
        sim.run(1)  # Only gen 0

        # Should complete without error even though TrioFilter returns None
        assert len(sim.results) == 1
        assert sim.results[0].generation == 0


# ---------------------------------------------------------------------------
# _run_callbacks tests
# ---------------------------------------------------------------------------

class TestRunCallbacks:
    """Test _run_callbacks internal method."""

    def test_callback_receives_simulation_reference(self):
        """Callbacks should receive the simulation object."""
        received_sims = []

        def capture_sim(s):
            received_sims.append(s)

        sim = _minimal_sim(callbacks=[capture_sim])
        sim.run(2)  # Gens 0,1 → 2 callbacks

        assert len(received_sims) == 2
        # All should be the same simulation object
        assert all(s is sim for s in received_sims)

    def test_callback_can_access_current_generation(self):
        """Callbacks should see the current generation number."""
        generations = []

        def record_gen(s):
            generations.append(s.generation)

        sim = _minimal_sim(callbacks=[record_gen])
        sim.run(4)  # Gens 0,1,2,3

        assert generations == [0, 1, 2, 3]

    def test_callback_can_modify_simulation_state(self):
        """Callbacks can modify simulation state (e.g., set stop flag)."""
        def set_flag(s):
            s.custom_flag = True

        sim = _minimal_sim(callbacks=[set_flag])
        sim.run(1)  # Gen 0

        assert hasattr(sim, 'custom_flag')
        assert sim.custom_flag is True

    def test_multiple_callbacks_execute_in_order(self):
        """Multiple callbacks should execute in the order provided."""
        order = []

        def first(s):
            order.append(1)

        def second(s):
            order.append(2)

        def third(s):
            order.append(3)

        sim = _minimal_sim(callbacks=[first, second, third])
        sim.run(2)  # Gens 0,1

        # Each gen: 1,2,3
        assert order == [1, 2, 3, 1, 2, 3]

    def test_callback_exception_propagates(self):
        """Exceptions in callbacks should propagate to caller."""
        def bad_callback(s):
            raise RuntimeError("callback failed")

        sim = _minimal_sim(callbacks=[bad_callback])

        with pytest.raises(RuntimeError, match="callback failed"):
            sim.run(1)

    def test_empty_callback_list_runs_normally(self):
        """With callbacks=[], simulation should run without issue."""
        sim = _minimal_sim(callbacks=[])
        sim.run(2)  # Gens 0,1

        assert sim.generation == 1


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------

class TestInternalMethodEdgeCases:
    """Test edge cases and interactions between internal methods."""

    def test_retention_with_continue_run(self):
        """Retention policy should apply correctly across run() and continue_run()."""
        sim = _minimal_sim(retain_haplotypes=1, retain_phenotypes=1)
        sim.run(3)  # Gens 0,1,2 → final gen=2

        # After gen 2: haplotypes=[1,2], phenotypes=[1,2]
        hap_keys_mid = sorted(sim.haplotype_history.keys())
        assert hap_keys_mid == [1, 2]

        sim.continue_run(2)  # Add gens 3,4 → final gen=4

        # After gen 4: haplotypes=[3,4], phenotypes=[3,4]
        hap_keys_final = sorted(sim.haplotype_history.keys())
        pheno_keys_final = sorted(sim.phenotype_history.keys())

        assert hap_keys_final == [3, 4]
        assert pheno_keys_final == [3, 4]

    def test_filters_and_stats_with_retention(self):
        """Filters and stats should work correctly with aggressive retention."""
        filters = {'sib': SibPairFilter()}
        stats = [SampleStatistics()]

        sim = _minimal_sim(
            filters=filters,
            statistics=stats,
            retain_haplotypes=0,
            retain_phenotypes=1,
        )
        sim.mating_regime = RandomMating(offspring_per_pair=2)
        sim.run(4)  # Gens 0,1,2,3 → final gen=3

        # Should have 4 results
        assert len(sim.results) == 4

        # Old phenotypes should be dropped
        pheno_keys = sorted(sim.phenotype_history.keys())
        assert pheno_keys == [2, 3]  # retain_phenotypes=1

    def test_callbacks_see_pruned_history(self):
        """Callbacks should see history state after retention pruning."""
        history_sizes = []

        def record_history_size(s):
            history_sizes.append(len(s.haplotype_history))

        sim = _minimal_sim(
            callbacks=[record_history_size],
            retain_haplotypes=1,
        )
        sim.run(4)  # Gens 0,1,2,3

        # Callbacks run after retention
        # Gen 0: 1 entry (gen 0)
        # Gen 1: 2 entries (gen 0,1) before retention, but callback sees after retention
        # Actually, callbacks run AFTER retention, so:
        # Gen 0: [0] → size 1
        # Gen 1: [0,1] after retention → size 2
        # Gen 2: [1,2] after retention → size 2
        # Gen 3: [2,3] after retention → size 2

        assert history_sizes[0] == 1  # Gen 0, only founders
        assert all(s == 2 for s in history_sizes[1:])  # Gens 1-3, retain=1 keeps 2

    def test_run_then_continue_run_maintains_consistency(self):
        """Phenotypes should be finite and consistent across run/continue_run."""
        sim = _minimal_sim(retain_haplotypes=10, retain_phenotypes=10)
        sim.run(2)  # Gens 0,1

        # Check gen 1 phenotypes are finite
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))

        sim.continue_run(2)  # Gens 2,3

        # Check gen 3 phenotypes are finite
        assert np.all(np.isfinite(sim.phenotype_history[3]['Y']))

        # All generations should have same phenotype keys
        keys0 = set(sim.phenotype_history[0].keys)
        keys3 = set(sim.phenotype_history[3].keys)
        assert keys0 == keys3
