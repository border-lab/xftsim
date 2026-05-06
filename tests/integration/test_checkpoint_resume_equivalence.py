"""
Integration tests for checkpoint-resume equivalence.

Tests that checkpointing and resuming produces valid, consistent results.
Note: Due to mate assignments not being checkpointed, results after resume
will differ from straight-through runs (different individuals mate). However,
statistical properties and validity of results should be preserved.

Key properties tested:
1. Checkpoint within a generation preserves exact state up to that point
2. RNG state is preserved for operations after mate assignment
3. Checkpoint roundtrip preserves all required state
4. Directory structure is correct after checkpoint
5. Resume from checkpoint correctly runs additional generations with valid results
6. Statistical properties are preserved across checkpoint boundary
"""
import numpy as np
import pytest
import tempfile
import shutil
import os

from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent,
    AggregationComponent, ParentComponent,
)
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_simple_sim(n=100, m=20, h2=0.5, seed=42, **kwargs):
    """Create a simple single-trait simulation."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m, p=0.5),
        seed=seed,
        **kwargs,
    )


def _make_bivariate_sim(n=100, m=20, h2=None, rg=0.3, seed=42, **kwargs):
    """Create a bivariate trait simulation."""
    if h2 is None:
        h2 = [0.5, 0.4]
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = MultivariateEffects.from_h2_rg(h2=h2, rg=rg, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add(['trait1.G', 'trait2.G'], MVGeneticComponent(eff))
    arch.add('trait1.E', NoiseComponent(variance=1.0 - h2[0]))
    arch.add('trait2.E', NoiseComponent(variance=1.0 - h2[1]))
    arch.add('trait1', AggregationComponent('trait1.G + trait1.E'))
    arch.add('trait2', AggregationComponent('trait2.G + trait2.E'))
    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m, p=0.5),
        seed=seed,
        **kwargs,
    )


def _make_vt_sim(n=100, m=20, h2=0.5, vt_weight=0.3, seed=42, **kwargs):
    """Create a vertical transmission simulation."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.5)))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent(f'Y.G + {vt_weight} * Y.VT + Y.E'))
    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m, p=0.5),
        seed=seed,
        **kwargs,
    )


class TestCheckpointResumeEquivalence:
    """Test that checkpoint-resume produces valid, statistically consistent results."""

    def test_checkpointed_generation_exact_match(self):
        """
        Phenotypes at the checkpointed generation should match exactly.
        This tests that saving and loading preserves existing state.
        """
        seed = 12345
        checkpoint_gen = 2

        # Run to checkpoint
        sim1 = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim1.run(checkpoint_gen + 1)  # gen 0, 1, 2
        pheno_at_checkpoint = sim1.phenotype_history[checkpoint_gen]['Y'].copy()
        hap_at_checkpoint = sim1.haplotype_history[checkpoint_gen].genotypes.copy()

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim1, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)

            # State at checkpoint should match exactly
            np.testing.assert_allclose(
                sim_resumed.phenotype_history[checkpoint_gen]['Y'],
                pheno_at_checkpoint,
                rtol=1e-10, atol=1e-10
            )
            np.testing.assert_array_equal(
                sim_resumed.haplotype_history[checkpoint_gen].genotypes,
                hap_at_checkpoint
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_resumed_phenotypes_valid(self):
        """
        After resume, phenotypes should be finite and have expected variance structure.
        Tests that resumed sim produces statistically valid results.
        """
        seed = 54321
        h2 = 0.6

        sim = _make_simple_sim(seed=seed, h2=h2, retain_haplotypes=3, retain_phenotypes=3)
        sim.run(2)  # gen 0, 1

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)  # gen 2, 3

            # Check phenotypes are valid
            pheno = sim_resumed.phenotypes
            assert 'Y' in pheno
            assert 'Y.G' in pheno
            assert 'Y.E' in pheno
            assert np.all(np.isfinite(pheno['Y']))
            assert np.all(np.isfinite(pheno['Y.G']))
            assert np.all(np.isfinite(pheno['Y.E']))

            # Check variance structure: genetic variance should be substantial
            var_g = np.var(pheno['Y.G'])
            var_e = np.var(pheno['Y.E'])
            assert var_g > 0.2  # Should have genetic variance
            assert var_e > 0.1  # Should have environmental variance
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_roundtrip_all_keys_present(self):
        """
        Save → load → verify all expected keys are present in the checkpoint dict.
        """
        sim = _make_simple_sim(seed=999, retain_haplotypes=2, retain_phenotypes=3)
        sim.run(3)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            checkpoint = load_simulation_checkpoint(tmpdir)

            # Check all expected keys
            expected_keys = {
                'architecture', 'generation', 'retain_haplotypes', 'retain_phenotypes',
                'rng', 'haplotype_history', 'phenotype_history', 'pedigree_history',
                'recombination_map', 'mating_regime',
            }
            assert set(checkpoint.keys()) == expected_keys

            # Check generation counter
            assert checkpoint['generation'] == 2

            # Check histories are dicts
            assert isinstance(checkpoint['haplotype_history'], dict)
            assert isinstance(checkpoint['phenotype_history'], dict)
            assert isinstance(checkpoint['pedigree_history'], dict)

            # Check RNG is a RandomState
            assert isinstance(checkpoint['rng'], np.random.RandomState)

            # Check retention settings
            assert checkpoint['retain_haplotypes'] == 2
            assert checkpoint['retain_phenotypes'] == 3
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_directory_structure(self):
        """
        Checkpoint directory should have correct structure:
        - meta.json
        - architecture/
        - haplotypes/
        - phenotypes/
        - pedigrees/
        - recombination_map.npz
        - rng_state.npz
        - history_keys.npz
        """
        sim = _make_simple_sim(seed=777, retain_haplotypes=2, retain_phenotypes=2)
        sim.run(3)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)

            # Check top-level files
            assert os.path.exists(os.path.join(tmpdir, 'meta.json'))
            assert os.path.exists(os.path.join(tmpdir, 'recombination_map.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'rng_state.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'history_keys.npz'))

            # Check directories
            assert os.path.isdir(os.path.join(tmpdir, 'architecture'))
            assert os.path.isdir(os.path.join(tmpdir, 'haplotypes'))
            assert os.path.isdir(os.path.join(tmpdir, 'phenotypes'))
            assert os.path.isdir(os.path.join(tmpdir, 'pedigrees'))

            # Check architecture.json exists
            assert os.path.exists(os.path.join(tmpdir, 'architecture', 'architecture.json'))

            # Check generation files exist
            hap_dir = os.path.join(tmpdir, 'haplotypes')
            pheno_dir = os.path.join(tmpdir, 'phenotypes')
            ped_dir = os.path.join(tmpdir, 'pedigrees')

            # Should have files for retained generations
            for gen in sim.haplotype_history.keys():
                assert os.path.exists(os.path.join(hap_dir, f'gen_{gen}.npz'))
            for gen in sim.phenotype_history.keys():
                assert os.path.exists(os.path.join(pheno_dir, f'gen_{gen}.npz'))
            for gen in sim.pedigree_history.keys():
                assert os.path.exists(os.path.join(ped_dir, f'gen_{gen}.npz'))
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_runs_additional_generations_correctly(self):
        """
        Resume from checkpoint and run additional generations.
        Verify that generation counter advances correctly and phenotypes are valid.
        """
        sim = _make_simple_sim(seed=333, retain_haplotypes=3, retain_phenotypes=3)
        sim.run(3)  # gen 0, 1, 2
        assert sim.generation == 2

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            assert sim_resumed.generation == 2

            # Run 3 more generations
            sim_resumed.continue_run(3)
            assert sim_resumed.generation == 5

            # Check phenotypes are finite and have variance
            pheno = sim_resumed.phenotypes
            assert 'Y' in pheno
            assert np.all(np.isfinite(pheno['Y']))
            assert np.var(pheno['Y']) > 0.1
        finally:
            shutil.rmtree(tmpdir)

    def test_bivariate_checkpoint_valid(self):
        """
        Test that bivariate trait simulation produces valid results after checkpoint-resume.
        """
        seed = 11111

        sim = _make_bivariate_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim.run(2)  # gen 0, 1

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)  # gen 2, 3

            # Check both traits are present and valid
            pheno = sim_resumed.phenotypes
            assert 'trait1' in pheno
            assert 'trait2' in pheno
            assert np.all(np.isfinite(pheno['trait1']))
            assert np.all(np.isfinite(pheno['trait2']))

            # Check variance structure
            var1 = np.var(pheno['trait1'])
            var2 = np.var(pheno['trait2'])
            assert var1 > 0.2
            assert var2 > 0.2

            # Check correlation exists (since rg > 0 in the architecture)
            corr = np.corrcoef(pheno['trait1.G'], pheno['trait2.G'])[0, 1]
            assert abs(corr) > 0.1  # Should have some genetic correlation
        finally:
            shutil.rmtree(tmpdir)

    def test_vertical_transmission_checkpoint_valid(self):
        """
        Test that vertical transmission simulation produces valid results after checkpoint-resume.
        VT components depend on parent phenotypes, testing cross-generation dependencies.
        """
        seed = 22222

        sim = _make_vt_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim.run(2)  # gen 0, 1

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)  # gen 2, 3

            # Check VT components are present and valid
            pheno = sim_resumed.phenotypes
            assert 'Y' in pheno
            assert 'Y.VT' in pheno
            assert 'Y.G' in pheno
            assert np.all(np.isfinite(pheno['Y']))
            assert np.all(np.isfinite(pheno['Y.VT']))

            # Check variance structure
            var_total = np.var(pheno['Y'])
            var_g = np.var(pheno['Y.G'])
            var_vt = np.var(pheno['Y.VT'])
            assert var_total > 0.3
            assert var_g > 0.1  # Genetic component should contribute
            assert var_vt > 0.05  # VT component should contribute
        finally:
            shutil.rmtree(tmpdir)

    def test_assortative_mating_checkpoint_valid(self):
        """
        Test that assortative mating simulation produces valid results after checkpoint-resume.
        """
        seed = 33333
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=2)

        sim = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim.mating_regime = mating
        sim.run(2)  # gen 0, 1

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)  # gen 2, 3

            # Check phenotypes are valid
            pheno = sim_resumed.phenotypes
            assert 'Y' in pheno
            assert np.all(np.isfinite(pheno['Y']))
            assert np.var(pheno['Y']) > 0.3

            # Check that pedigree was created (assortative mating produced offspring)
            assert sim_resumed.generation in sim_resumed.pedigree_history
            ped = sim_resumed.pedigree_history[sim_resumed.generation]
            assert len(ped.maternal_idx) > 0
            assert len(ped.paternal_idx) > 0
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_with_retention_policy(self):
        """
        Test that checkpoint saves only retained generations and resume works correctly.
        """
        sim = _make_simple_sim(seed=444, retain_haplotypes=1, retain_phenotypes=2)
        sim.run(5)  # gen 0-4

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            checkpoint = load_simulation_checkpoint(tmpdir)

            # Should only have retained generations
            # retain_haplotypes=1: should have gen 3, 4
            # retain_phenotypes=2: should have gen 2, 3, 4
            assert len(checkpoint['haplotype_history']) <= 2
            assert len(checkpoint['phenotype_history']) <= 3

            # Most recent generation should be present
            assert 4 in checkpoint['haplotype_history']
            assert 4 in checkpoint['phenotype_history']

            # Resume should work
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            assert sim_resumed.generation == 4
            sim_resumed.continue_run(1)
            assert sim_resumed.generation == 5
            assert np.all(np.isfinite(sim_resumed.phenotypes['Y']))
        finally:
            shutil.rmtree(tmpdir)

    def test_multiple_checkpoint_cycles(self):
        """
        Test that multiple checkpoint-resume cycles produce valid results.
        """
        seed = 55555

        # Multiple checkpoint cycles
        sim = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim.run(2)  # gen 0, 1

        tmpdir1 = tempfile.mkdtemp()
        try:
            # First checkpoint
            save_simulation_checkpoint(sim, tmpdir1)
            sim = NSimulation.from_checkpoint(tmpdir1)
            sim.continue_run(2)  # gen 2, 3

            tmpdir2 = tempfile.mkdtemp()
            try:
                # Second checkpoint
                save_simulation_checkpoint(sim, tmpdir2)
                sim = NSimulation.from_checkpoint(tmpdir2)
                sim.continue_run(2)  # gen 4, 5

                # Check final state is valid
                assert sim.generation == 5
                pheno = sim.phenotypes
                assert 'Y' in pheno
                assert np.all(np.isfinite(pheno['Y']))
                assert np.var(pheno['Y']) > 0.2

                # Check we have history
                assert len(sim.phenotype_history) > 0
                assert len(sim.haplotype_history) > 0
            finally:
                shutil.rmtree(tmpdir2)
        finally:
            shutil.rmtree(tmpdir1)

    def test_checkpoint_preserves_haplotype_structure(self):
        """
        Test that haplotype structure and allele frequencies are valid after checkpoint-resume.
        """
        seed = 66666

        sim = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim.run(2)
        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)

            # Check haplotype structure
            hap = sim_resumed.haplotypes
            assert hap.n > 0
            assert hap.m > 0
            assert hap.genotypes.shape == (hap.n, hap.m, 2)
            assert hap.genotypes.dtype == np.int8

            # Check allele frequencies are valid
            af = hap.recompute_af()
            assert np.all(af >= 0.0)
            assert np.all(af <= 1.0)

            # Check genotypes are binary
            assert np.all((hap.genotypes == 0) | (hap.genotypes == 1))
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_pedigree_equivalence(self):
        """
        Test that pedigree information is identical after checkpoint-resume.
        """
        seed = 77777

        # Straight-through run
        sim1 = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim1.run(4)
        ped_straight = sim1.pedigree_history[3]
        mat_straight = ped_straight.maternal_idx.copy()
        pat_straight = ped_straight.paternal_idx.copy()

        # Checkpoint-resume run
        sim2 = _make_simple_sim(seed=seed, retain_haplotypes=5, retain_phenotypes=5)
        sim2.run(2)
        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim2, tmpdir)
            sim2_resumed = NSimulation.from_checkpoint(tmpdir)
            sim2_resumed.continue_run(2)
            ped_resumed = sim2_resumed.pedigree_history[3]
            mat_resumed = ped_resumed.maternal_idx.copy()
            pat_resumed = ped_resumed.paternal_idx.copy()

            # Pedigree indices should be identical
            np.testing.assert_array_equal(mat_resumed, mat_straight)
            np.testing.assert_array_equal(pat_resumed, pat_straight)
        finally:
            shutil.rmtree(tmpdir)
