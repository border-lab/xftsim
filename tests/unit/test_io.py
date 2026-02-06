"""
I/O round-trip tests for haplotype save/load.
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta
from xftsim.io import save_haplotypes_npz, load_haplotypes_npz

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
