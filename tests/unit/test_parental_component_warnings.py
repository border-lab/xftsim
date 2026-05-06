"""
Unit tests for _ParentalComponent warning paths.

Tests:
1. Gen 0 without founder_component → warns + returns zeros
2. Gen 0 with founder_component → delegates
3. Missing prev gen phenotypes → warns + returns zeros
4. Missing phenotype_name → raises ValueError
5. MotherComponent normal operation
6. FatherComponent normal operation
7. ParentComponent midparent average
8. repr
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, NPhenotypeArray, PedigreeArray, DenseHaplotypeArray
from xftsim.arch import (
    Architecture, ArchNode,
    MotherComponent, FatherComponent, ParentComponent,
    NoiseComponent,
)


def _make_hap(n=10, m=5, seed=42, generation=0):
    rng = np.random.RandomState(seed)
    from xftsim.struct import VariantMeta
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, generation=generation)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


def _make_pheno(n, keys_values, generation=0):
    sm = SampleMeta(iid=np.arange(n), generation=generation)
    pheno = NPhenotypeArray(samples=sm)
    for k, v in keys_values.items():
        pheno._values[k] = np.asarray(v, dtype=np.float64)
    return pheno


def _make_ped(n_offspring, parent_n):
    sm = SampleMeta(iid=np.arange(n_offspring), fid=np.repeat(np.arange(n_offspring // 2), 2)[:n_offspring], generation=1)
    n_pairs = n_offspring // 2
    return PedigreeArray(
        offspring_samples=sm,
        maternal_idx=np.repeat(np.arange(n_pairs), 2)[:n_offspring],
        paternal_idx=np.repeat(np.arange(n_pairs, 2 * n_pairs), 2)[:n_offspring],
        parent_n=parent_n,
    )


class TestParentalComponentGen0:
    def test_gen0_no_founder_warns_zeros(self):
        """Gen 0 without founder_component → warning + zeros."""
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])
        hap = _make_hap(n=10)
        pheno = _make_pheno(10, {'Y': np.arange(10, dtype=float)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = comp.compute(node, hap, pheno, generation=0, phenotype_history={}, pedigree_history={})
        assert len(w) == 1
        assert 'returning zeros' in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_gen0_with_founder_delegates(self):
        """Gen 0 with founder_component → delegates to founder."""
        founder = NoiseComponent(variance=1.0)
        comp = MotherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])
        hap = _make_hap(n=10)
        pheno = _make_pheno(10, {})
        result = comp.compute(node, hap, pheno, generation=0,
                              phenotype_history={}, pedigree_history={},
                              rng=np.random.RandomState(42))
        assert result.shape == (10,)
        # Noise should be non-zero (with high probability)
        assert not np.allclose(result, 0)


class TestParentalComponentPrunedHistory:
    def test_pruned_phenotype_history_warns_zeros(self):
        """If prev_gen not in phenotype_history, warns and returns zeros."""
        comp = FatherComponent('Y')
        node = ArchNode(outputs=['Y.VTf'], component=comp, inputs=[])
        hap = _make_hap(n=10, generation=2)
        pheno = _make_pheno(10, {'Y': np.ones(10)}, generation=2)
        ped = _make_ped(10, parent_n=10)
        # pedigree at gen 2, but phenotype_history only has gen 2, not gen 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = comp.compute(node, hap, pheno, generation=2,
                                  phenotype_history={2: pheno},
                                  pedigree_history={2: ped})
        assert len(w) == 1
        assert 'not in phenotype_history' in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(10))


class TestParentalComponentMissingKey:
    def test_missing_phenotype_name_raises(self):
        """If phenotype_name not in prev_gen, raises ValueError."""
        comp = MotherComponent('NONEXISTENT')
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])
        hap = _make_hap(n=10, generation=1)
        pheno_prev = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=0)
        pheno_cur = _make_pheno(10, {'Y': np.ones(10)}, generation=1)
        ped = _make_ped(10, parent_n=10)
        with pytest.raises(ValueError, match="not found"):
            comp.compute(node, hap, pheno_cur, generation=1,
                         phenotype_history={0: pheno_prev, 1: pheno_cur},
                         pedigree_history={1: ped})


class TestMotherFatherParent:
    def _setup(self):
        parent_vals = np.arange(10, dtype=float)
        pheno_prev = _make_pheno(10, {'Y': parent_vals}, generation=0)
        pheno_cur = _make_pheno(10, {'Y': np.zeros(10)}, generation=1)
        ped = _make_ped(10, parent_n=10)
        hap = _make_hap(n=10, generation=1)
        return pheno_prev, pheno_cur, ped, hap

    def test_mother_extracts_maternal(self):
        pheno_prev, pheno_cur, ped, hap = self._setup()
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[])
        result = comp.compute(node, hap, pheno_cur, generation=1,
                              phenotype_history={0: pheno_prev, 1: pheno_cur},
                              pedigree_history={1: ped})
        expected = pheno_prev['Y'][ped.maternal_idx]
        np.testing.assert_array_equal(result, expected)

    def test_father_extracts_paternal(self):
        pheno_prev, pheno_cur, ped, hap = self._setup()
        comp = FatherComponent('Y')
        node = ArchNode(outputs=['Y.f'], component=comp, inputs=[])
        result = comp.compute(node, hap, pheno_cur, generation=1,
                              phenotype_history={0: pheno_prev, 1: pheno_cur},
                              pedigree_history={1: ped})
        expected = pheno_prev['Y'][ped.paternal_idx]
        np.testing.assert_array_equal(result, expected)

    def test_parent_is_midparent(self):
        pheno_prev, pheno_cur, ped, hap = self._setup()
        comp = ParentComponent('Y')
        node = ArchNode(outputs=['Y.p'], component=comp, inputs=[])
        result = comp.compute(node, hap, pheno_cur, generation=1,
                              phenotype_history={0: pheno_prev, 1: pheno_cur},
                              pedigree_history={1: ped})
        expected = 0.5 * (pheno_prev['Y'][ped.maternal_idx] + pheno_prev['Y'][ped.paternal_idx])
        np.testing.assert_allclose(result, expected)


class TestParentalRepr:
    def test_mother_repr(self):
        r = repr(MotherComponent('Y'))
        assert 'MotherComponent' in r
        assert "'Y'" in r

    def test_father_repr(self):
        r = repr(FatherComponent('Y'))
        assert 'FatherComponent' in r

    def test_parent_repr(self):
        r = repr(ParentComponent('Y'))
        assert 'ParentComponent' in r
