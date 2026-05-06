"""
Tests for the grouping operator (|), grouped noise, and CNoiseComponent.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation, TestMeta

from xftsim.struct import SampleMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.arch import (
    Architecture, NoiseComponent, CNoiseComponent, AggregationComponent, ArchNode,
)
from xftsim.parser import parse_formula


class TestGroupedNoise:
    @pytest.fixture
    def hap_with_fam(self):
        """Haplotypes with structured families (5 per family)."""
        n = 100
        m = 10
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        fid = np.repeat(np.arange(20), 5)  # 20 families of 5
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex)
        return DenseHaplotypeArray(genotypes=geno, samples=samples)

    def test_noise_bare_individual_specific(self, hap_with_fam):
        """Noise without grouping: all values should be different."""
        arch = Architecture()
        arch.add('E', NoiseComponent(1.0))
        pheno = arch.compute(hap_with_fam, rng=np.random.RandomState(42))
        vals = pheno['E']
        # With 100 iid draws, extremely unlikely any two are exactly equal
        assert len(np.unique(vals)) == len(vals)

    def test_noise_fid_shared_within_family(self, hap_with_fam):
        """noise | FID: same value within each family."""
        arch = Architecture()
        arch.add('E', NoiseComponent(1.0), grouping='FID')
        pheno = arch.compute(hap_with_fam, rng=np.random.RandomState(42))
        vals = pheno['E']
        fids = hap_with_fam.samples.fid
        # All members of each family should have the same value
        for fam in np.unique(fids):
            fam_vals = vals[fids == fam]
            assert np.all(fam_vals == fam_vals[0])

    def test_grouped_variance_correct(self, hap_with_fam):
        """Grouped noise should have approximately correct variance (across groups)."""
        arch = Architecture()
        arch.add('E', NoiseComponent(2.0), grouping='FID')
        pheno = arch.compute(hap_with_fam, rng=np.random.RandomState(99))
        vals = pheno['E']
        fids = hap_with_fam.samples.fid
        # Get one value per family
        unique_fids = np.unique(fids)
        group_vals = np.array([vals[fids == f][0] for f in unique_fids])
        # Variance should be roughly 2.0 (20 groups, so imprecise)
        assert 0.3 < np.var(group_vals) < 6.0

    def test_noise_mother_grouping(self):
        """noise | mother groups by maternal_idx."""
        from xftsim.struct import PedigreeArray
        n = 20
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        # Simulate pedigree: 10 pairs, 2 offspring each
        maternal_idx = np.repeat(np.arange(10), 2)
        paternal_idx = np.repeat(np.arange(10, 20), 2)
        offspring_samples = SampleMeta(iid=np.arange(n), sex=sex)
        ped = PedigreeArray(offspring_samples=offspring_samples,
                           maternal_idx=maternal_idx,
                           paternal_idx=paternal_idx,
                           parent_n=20)

        arch = Architecture()
        arch.add('E', NoiseComponent(1.0), grouping='mother')
        pheno = arch.compute(hap, rng=np.random.RandomState(42),
                            pedigree_history={1: ped}, generation=1)
        vals = pheno['E']
        # Siblings (same maternal_idx) should share the same value
        for i in range(0, 20, 2):
            assert vals[i] == vals[i + 1]

    def test_noise_father_grouping(self):
        """noise | father groups by paternal_idx."""
        from xftsim.struct import PedigreeArray
        n = 20
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        maternal_idx = np.repeat(np.arange(10), 2)
        paternal_idx = np.repeat(np.arange(10, 20), 2)
        offspring_samples = SampleMeta(iid=np.arange(n), sex=sex)
        ped = PedigreeArray(offspring_samples=offspring_samples,
                           maternal_idx=maternal_idx,
                           paternal_idx=paternal_idx,
                           parent_n=20)

        arch = Architecture()
        arch.add('E', NoiseComponent(1.0), grouping='father')
        pheno = arch.compute(hap, rng=np.random.RandomState(42),
                            pedigree_history={1: ped}, generation=1)
        vals = pheno['E']
        for i in range(0, 20, 2):
            assert vals[i] == vals[i + 1]

    def test_extra_field_grouping(self):
        """Grouping by extra field on SampleMeta."""
        n = 50
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        batch = np.repeat([0, 1, 2, 3, 4], 10)
        samples = SampleMeta(iid=np.arange(n), extra={'batch': batch})
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        arch = Architecture()
        arch.add('E', NoiseComponent(1.0), grouping='batch')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        vals = pheno['E']
        # Same batch should have same value
        for b in range(5):
            batch_vals = vals[batch == b]
            assert np.all(batch_vals == batch_vals[0])


class TestCNoiseComponent:
    @pytest.fixture
    def hap(self):
        return TestSimulation.founder_haplotypes(n=500, m=10)

    def test_cnoise_shape(self, hap):
        """cnoise should produce correct output shapes."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        arch = Architecture()
        arch.add(['a', 'b'], CNoiseComponent(cov=cov))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert pheno['a'].shape == (500,)
        assert pheno['b'].shape == (500,)

    def test_cnoise_covariance(self, hap):
        """cnoise should produce approximately correct covariance (statistical)."""
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        arch = Architecture()
        arch.add(['x', 'y'], CNoiseComponent(cov=cov))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        data = np.column_stack([pheno['x'], pheno['y']])
        obs_cov = np.cov(data, rowvar=False)
        # With n=500, tolerances need to be loose
        np.testing.assert_allclose(obs_cov, cov, atol=0.3)

    def test_cnoise_fid_grouping(self):
        """cnoise | FID: same vector within each family."""
        n = 100
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        fid = np.repeat(np.arange(20), 5)
        samples = SampleMeta(iid=np.arange(n), fid=fid)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        arch = Architecture()
        arch.add(['a', 'b'], CNoiseComponent(cov=cov), grouping='FID')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        for fam in np.unique(fid):
            mask = fid == fam
            assert np.all(pheno['a'][mask] == pheno['a'][mask][0])
            assert np.all(pheno['b'][mask] == pheno['b'][mask][0])


class TestParserPipeSyntax:
    def test_noise_pipe_fid(self):
        """Parser should handle noise(0.5) | FID."""
        nodes = parse_formula("E ~ noise(0.5) | FID")
        assert len(nodes) == 1
        assert nodes[0].grouping == 'FID'
        assert isinstance(nodes[0].component, NoiseComponent)

    def test_cnoise_via_formula(self):
        """Parser should handle (a, b) ~ cnoise(cov=[[1,0.3],[0.3,1]])."""
        nodes = parse_formula("(a, b) ~ cnoise(cov=[[1,0.3],[0.3,1]])")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, CNoiseComponent)
        assert nodes[0].component.k == 2
        assert nodes[0].outputs == ['a', 'b']
        assert nodes[0].grouping is None

    def test_cnoise_grouped_via_formula(self):
        """Parser should handle (a, b) ~ cnoise(cov=[[1,0.3],[0.3,1]]) | FID."""
        nodes = parse_formula("(a, b) ~ cnoise(cov=[[1,0.3],[0.3,1]]) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, CNoiseComponent)
        assert nodes[0].grouping == 'FID'

    def test_invalid_grouping_on_genetic(self):
        """Grouping on genetic should raise error."""
        from xftsim.effect import AdditiveEffects
        eff = AdditiveEffects.from_h2(0.5, 10)
        with pytest.raises(ValueError, match="does not accept"):
            parse_formula("G ~ genetic(eff) | FID", effects={'eff': eff})
