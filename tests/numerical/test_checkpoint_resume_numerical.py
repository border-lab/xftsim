"""
Numerical test: checkpoint-resume produces valid simulation state.

Tests:
1. Resumed simulation continues from correct generation
2. Resumed simulation produces valid phenotypes
3. Resumed simulation has correct history lengths
4. Statistics collected during resumed run are valid
"""
import numpy as np
import pytest
import tempfile
import shutil

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics
from xftsim.io import save_simulation_checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=200, m=20, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, **kwargs,
    )


class TestCheckpointResumeNumerical:
    def test_resume_continues_from_correct_generation(self):
        """Resumed sim should continue from the saved generation."""
        sim = _make_sim()
        sim.run(3)
        assert sim.generation == 2

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            assert sim_resumed.generation == 2
            sim_resumed.continue_run(2)
            assert sim_resumed.generation == 4
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_produces_valid_phenotypes(self):
        """Phenotypes after resume should be finite and have genetic variance."""
        sim = _make_sim()
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(3)
            pheno = sim_resumed.phenotypes
            assert 'Y' in pheno
            assert 'Y.G' in pheno
            assert 'Y.E' in pheno
            assert np.all(np.isfinite(pheno['Y']))
            assert np.var(pheno['Y.G']) > 0.01
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_history_lengths(self):
        """History dicts should contain correct generations after resume."""
        sim = _make_sim(retain_haplotypes=2, retain_phenotypes=2)
        sim.run(3)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(2)
            # After retention, should have at most retain+1 entries
            assert len(sim_resumed.haplotype_history) <= 3
            assert len(sim_resumed.phenotype_history) <= 3
            # Final generation should be present
            assert sim_resumed.generation in sim_resumed.haplotype_history
            assert sim_resumed.generation in sim_resumed.phenotype_history
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_with_statistics(self):
        """Statistics should be collected correctly after resume."""
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(
                tmpdir, statistics=[SampleStatistics()],
            )
            sim_resumed.continue_run(2)
            # Should have results for continued generations
            assert len(sim_resumed.results) > 0
            last = sim_resumed.results[-1]
            stat = last.statistics.get('SampleStatistics')
            assert stat is not None
            assert 'var' in stat
            assert 'keys' in stat
            # Y variance should be positive
            y_idx = stat['keys'].index('Y')
            assert stat['var'][y_idx] > 0.1
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_allele_frequencies_valid(self):
        """AF after resumed simulation should be in [0, 1]."""
        sim = _make_sim()
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            sim_resumed = NSimulation.from_checkpoint(tmpdir)
            sim_resumed.continue_run(3)
            af = sim_resumed.haplotypes.recompute_af()
            assert np.all(af >= 0.0)
            assert np.all(af <= 1.0)
        finally:
            shutil.rmtree(tmpdir)
