"""
I/O round-trip tests for haplotype and phenotype save/load.
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta, NPhenotypeArray
from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestHaplotypeIO:
    def test_roundtrip_genotypes(self, tmp_path):
        """Save and load should produce identical genotypes."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_equal(loaded.genotypes, hap.genotypes)

    def test_roundtrip_metadata(self, tmp_path):
        """Save and load should preserve sample and variant metadata."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_equal(loaded.iid, hap.iid)
        np.testing.assert_array_equal(loaded.fid, hap.fid)
        np.testing.assert_array_equal(loaded.sex, hap.sex)
        np.testing.assert_array_equal(loaded.vid, hap.vid)
        assert loaded.generation == hap.generation

    def test_roundtrip_with_af(self, tmp_path):
        """Allele frequencies should survive round-trip."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_almost_equal(loaded.af, hap.af)


class TestPhenotypeIO:
    def _make_pheno(self, n=50):
        samples = SampleMeta(iid=np.arange(n))
        pheno = NPhenotypeArray(samples=samples)
        rng = np.random.RandomState(42)
        pheno['Y.G'] = rng.randn(n)
        pheno['Y.E'] = rng.randn(n)
        pheno['Y'] = pheno['Y.G'] + pheno['Y.E']
        return pheno

    def test_roundtrip_values(self, tmp_path):
        """Save and load should produce identical phenotype values."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        for key in pheno.keys:
            np.testing.assert_array_equal(loaded[key], pheno[key])

    def test_roundtrip_keys(self, tmp_path):
        """All keys should be preserved."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        assert set(loaded.keys) == set(pheno.keys)

    def test_roundtrip_samples(self, tmp_path):
        """Sample metadata should be preserved."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        np.testing.assert_array_equal(loaded.samples.iid, pheno.samples.iid)

    def test_roundtrip_from_simulation(self, tmp_path):
        """Phenotypes from a real simulation should round-trip."""
        from xftsim.nsim import NSimulation
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap

        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mate, recombination_map=rmap, seed=42,
        )
        sim.run(1)
        pheno = sim.phenotype_history[0]

        path = str(tmp_path / "sim_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)

        for key in pheno.keys:
            np.testing.assert_array_almost_equal(loaded[key], pheno[key])
