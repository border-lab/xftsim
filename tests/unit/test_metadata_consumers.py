"""Unit tests for components that read per-individual metadata (SampleMeta.extra):
the meta() architecture builtin, MetadataFilter, MetadataSummary,
SampleStatistics(filter_name=...), GroupedMating, and metadata I/O."""
import json
import warnings

import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray, DenseHaplotypeArray
from xftsim.arch import Architecture, MetadataComponent
from xftsim.filters import MetadataFilter, SubsetView
from xftsim.stats import MetadataSummary, SampleStatistics
from xftsim.mate import (
    RandomMating, LinearAssortativeMating, BatchedMating, GroupedMating,
)
from xftsim import io as xio


def _samples(n=12):
    return SampleMeta(
        iid=np.arange(n),
        extra={
            'population': np.array(['A', 'B', 'C'])[np.arange(n) % 3].astype(object),
            'ancestry_A': (np.arange(n) % 3 == 0).astype(float),
            'deme': np.arange(n) // 6,
        },
    )


def _hap(samples, m=4):
    geno = np.zeros((samples.n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=samples)


# ---------------------------------------------------------------------------
# meta() component
# ---------------------------------------------------------------------------

class TestMetadataComponent:
    def test_parse_and_compute(self):
        arch = Architecture.from_formula("""
        Y.A ~ meta(ancestry_A)
        Y ~ 2*Y.A
        """)
        pheno = arch.compute(_hap(_samples()), rng=np.random.RandomState(0))
        np.testing.assert_array_equal(pheno['Y.A'], _samples().extra['ancestry_A'])
        np.testing.assert_array_equal(pheno['Y'], 2 * pheno['Y.A'])

    def test_metadata_keys(self):
        assert MetadataComponent('ancestry_A').metadata_keys == ('ancestry_A',)

    def test_non_numeric_rejected(self):
        arch = Architecture.from_formula("Y.P ~ meta(population)")
        with pytest.raises(TypeError, match="numeric"):
            arch.compute(_hap(_samples()), rng=np.random.RandomState(0))

    def test_missing_field_explains_rules(self):
        arch = Architecture.from_formula("Y.A ~ meta(nope)")
        with pytest.raises(KeyError, match="metadata_rules"):
            arch.compute(_hap(_samples()), rng=np.random.RandomState(0))

    def test_bad_argument(self):
        with pytest.raises(ValueError, match="meta\\(\\) requires"):
            Architecture.from_formula("Y.A ~ meta(a, b)")

    def test_rejects_grouping(self):
        with pytest.raises(ValueError, match="does not accept"):
            Architecture.from_formula("Y.A ~ meta(ancestry_A) | deme")

    def test_architecture_roundtrip(self, tmp_path):
        arch = Architecture.from_formula("""
        Y.A ~ meta(ancestry_A)
        Y ~ Y.A
        """)
        xio.save_architecture(arch, str(tmp_path))
        back = xio.load_architecture(str(tmp_path))
        comp = [n.component for n in back.nodes if n.outputs == ['Y.A']][0]
        assert isinstance(comp, MetadataComponent)
        assert comp.key == 'ancestry_A'


# ---------------------------------------------------------------------------
# Filters and statistics
# ---------------------------------------------------------------------------

def _history():
    s = _samples()
    pheno = PhenotypeArray(s, {'Y': np.arange(s.n, dtype=float)})
    return {0: pheno}


class TestMetadataFilter:
    def test_values(self):
        view = MetadataFilter('population', values=['A', 'C']).apply(0, _history(), {})
        assert isinstance(view, SubsetView)
        np.testing.assert_array_equal(view.indices, [0, 2, 3, 5, 6, 8, 9, 11])
        assert set(view.phenotypes.samples.extra['population']) == {'A', 'C'}
        np.testing.assert_array_equal(view.phenotypes['Y'], view.indices)

    def test_predicate(self):
        view = MetadataFilter('ancestry_A', predicate=lambda a: a > 0.5).apply(
            0, _history(), {})
        np.testing.assert_array_equal(view.indices, [0, 3, 6, 9])

    def test_predicate_bad_shape(self):
        filt = MetadataFilter('ancestry_A', predicate=lambda a: a[:2] > 0)
        with pytest.raises(ValueError, match="shape"):
            filt.apply(0, _history(), {})

    def test_exactly_one_selector(self):
        with pytest.raises(ValueError):
            MetadataFilter('population')
        with pytest.raises(ValueError):
            MetadataFilter('population', values=['A'], predicate=lambda a: a)

    def test_missing_field_returns_none(self):
        assert MetadataFilter('nope', values=[1]).apply(0, _history(), {}) is None

    def test_metadata_keys(self):
        assert MetadataFilter('deme', values=[0]).metadata_keys == ('deme',)


class TestMetadataSummary:
    def test_all_fields(self):
        out = MetadataSummary().estimate(_history(), {}, 0)
        assert out['ancestry_A']['mean'] == pytest.approx(1 / 3)
        assert out['ancestry_A']['var'] == pytest.approx(2 / 9)
        assert out['population']['counts'] == {'A': 4, 'B': 4, 'C': 4}
        assert out['deme']['mean'] == pytest.approx(0.5)
        assert out['deme']['n'] == 12

    def test_stratified(self):
        out = MetadataSummary(keys=['ancestry_A'], by='population').estimate(
            _history(), {}, 0)
        assert set(out) == {'A', 'B', 'C'}
        assert out['A']['ancestry_A']['mean'] == 1.0
        assert out['B']['ancestry_A']['mean'] == 0.0
        assert out['A']['ancestry_A']['n'] == 4

    def test_missing_key(self):
        with pytest.raises(KeyError, match="nope"):
            MetadataSummary(keys=['nope']).estimate(_history(), {}, 0)

    def test_unsortable_labels(self):
        s = SampleMeta(iid=np.arange(3),
                       extra={'g': np.array([None, 'a', 'b'], dtype=object),
                              'x': np.zeros(3)})
        hist = {0: PhenotypeArray(s, {})}
        with pytest.raises(TypeError, match="cannot be sorted"):
            MetadataSummary(keys=['x'], by='g').estimate(hist, {}, 0)

    def test_metadata_keys(self):
        assert MetadataSummary(keys=['a'], by='b').metadata_keys == ('a', 'b')
        assert MetadataSummary().metadata_keys == ()


class TestSampleStatisticsOnView:
    def test_filter_name(self):
        hist = _history()
        view = MetadataFilter('population', values=['A']).apply(0, hist, {})
        full = SampleStatistics().estimate(hist, {}, 0)
        sub = SampleStatistics(filter_name='popA').estimate(hist, {'popA': view}, 0)
        assert sub['var'][0] == pytest.approx(np.var([0, 3, 6, 9], ddof=1))
        assert sub['var'][0] != full['var'][0]

    def test_missing_view(self):
        assert SampleStatistics(filter_name='x').estimate(_history(), {}, 0) is None


# ---------------------------------------------------------------------------
# Mating
# ---------------------------------------------------------------------------

def _mating_samples(n=60):
    sex = np.tile([0, 1], n // 2)
    deme = np.repeat(['north', 'south', 'east'], n // 3)
    return SampleMeta(iid=np.arange(n), sex=sex, extra={'deme': deme})


class TestGroupedMating:
    def test_no_between_group_pairs(self):
        s = _mating_samples()
        a = GroupedMating(RandomMating(), by='deme').mate(
            s, rng=np.random.RandomState(0))
        deme = s.extra['deme']
        np.testing.assert_array_equal(deme[a.maternal_idx], deme[a.paternal_idx])
        assert a.n_offspring == s.n
        assert set(deme[a.maternal_idx]) == {'north', 'south', 'east'}

    def test_family_ids_unique_across_groups(self):
        s = _mating_samples()
        a = GroupedMating(RandomMating(offspring_per_pair=3), by='deme').mate(
            s, rng=np.random.RandomState(0))
        fam_parents = {}
        for f, m, p in zip(a.offspring_samples.fid, a.maternal_idx, a.paternal_idx):
            fam_parents.setdefault(f, set()).add((m, p))
        assert all(len(v) == 1 for v in fam_parents.values())
        assert len(fam_parents) == 30
        np.testing.assert_array_equal(np.bincount(a.offspring_samples.fid), 3)

    def test_single_sex_group_skipped(self):
        sex = np.array([0, 1, 0, 1, 0, 0])
        deme = np.array([1, 1, 1, 1, 2, 2])
        s = SampleMeta(iid=np.arange(6), sex=sex, extra={'deme': deme})
        with pytest.warns(UserWarning, match="lacks one sex"):
            a = GroupedMating(RandomMating(), by='deme').mate(
                s, rng=np.random.RandomState(0))
        assert set(a.maternal_idx) | set(a.paternal_idx) <= {0, 1, 2, 3}

    def test_all_groups_single_sex(self):
        s = SampleMeta(iid=np.arange(4), sex=np.array([0, 0, 1, 1]),
                       extra={'deme': np.array([1, 1, 2, 2])})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match="both sexes"):
                GroupedMating(RandomMating(), by='deme').mate(
                    s, rng=np.random.RandomState(0))

    def test_rejects_float_field(self):
        s = _samples()
        s = SampleMeta(iid=s.iid, extra={'ancestry_A': s.extra['ancestry_A']})
        with pytest.raises(TypeError, match="floating"):
            GroupedMating(RandomMating(), by='ancestry_A').mate(
                s, rng=np.random.RandomState(0))

    def test_unsortable_labels(self):
        s = SampleMeta(iid=np.arange(4),
                       extra={'deme': np.array([None, 'a', None, 'a'], dtype=object)})
        with pytest.raises(TypeError, match="cannot be sorted"):
            GroupedMating(RandomMating(), by='deme').mate(
                s, rng=np.random.RandomState(0))

    def test_missing_field(self):
        with pytest.raises(KeyError, match="metadata_rules"):
            GroupedMating(RandomMating(), by='deme').mate(
                SampleMeta(iid=np.arange(4)), rng=np.random.RandomState(0))

    def test_metadata_keys_and_serialization(self):
        reg = GroupedMating(BatchedMating(RandomMating(3), max_batch_size=10),
                            by='deme')
        assert reg.metadata_keys == ('deme',)
        config = xio._serialize_mating_regime(reg)
        json.dumps(config)
        back = xio._deserialize_mating_regime(config)
        assert isinstance(back, GroupedMating) and back.by == 'deme'
        assert isinstance(back.regime, BatchedMating)
        assert back.regime.regime.offspring_per_pair == 3


class TestBatchedMatingMerge:
    def test_matches_pre_refactor_output(self):
        """Golden values from BatchedMating before the merge refactor."""
        s = SampleMeta(iid=np.arange(20))
        a = BatchedMating(RandomMating(offspring_per_pair=3), max_batch_size=7).mate(
            s, rng=np.random.RandomState(4))
        np.testing.assert_array_equal(a.maternal_idx, np.repeat(
            [18, 4, 0, 2, 12, 10, 14, 8, 16], 3))
        np.testing.assert_array_equal(a.paternal_idx, np.repeat(
            [19, 13, 3, 9, 17, 15, 5, 1, 7], 3))
        np.testing.assert_array_equal(a.offspring_samples.fid, np.repeat(np.arange(9), 3))
        np.testing.assert_array_equal(a.offspring_samples.sex, np.tile([0, 1, 0], 9))

    def test_nested_wrappers_keep_inner_families(self):
        """Nested wrappers used to assume 2 offspring per pair for fid/sex."""
        s = _mating_samples()
        inner = GroupedMating(RandomMating(offspring_per_pair=3), by='deme')
        a = BatchedMating(inner, max_batch_size=1000).mate(
            s, rng=np.random.RandomState(0))
        np.testing.assert_array_equal(np.bincount(a.offspring_samples.fid), 3)
        np.testing.assert_array_equal(a.offspring_samples.sex,
                                      np.tile([0, 1, 0], 30))


class TestConstantTraitAssortment:
    def test_constant_trait_falls_back_to_random(self):
        """A constant composite used to pair adjacent full siblings."""
        n = 40
        s = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
        pheno = PhenotypeArray(s, {'A': np.full(n, 1 / 3)})
        a = LinearAssortativeMating(['A'], r=0.9).mate(
            s, rng=np.random.RandomState(0), phenotypes=pheno)
        siblings = s.fid[a.maternal_idx] == s.fid[a.paternal_idx]
        assert siblings.mean() < 0.5


# ---------------------------------------------------------------------------
# npz I/O of sample metadata
# ---------------------------------------------------------------------------

class TestSampleMetadataIO:
    def _rich_samples(self):
        n = 6
        return SampleMeta(
            iid=np.arange(n), fid=np.arange(n) // 2, generation=4,
            extra={
                'population': np.array(['YRI', 'CEU'] * 3, dtype=object),
                'ancestry_YRI': np.linspace(0, 1, n),
                'deme': np.arange(n, dtype=np.int32),
                'flag': np.array([True, False] * 3),
            },
        )

    def _assert_same(self, a, b):
        np.testing.assert_array_equal(a.iid, b.iid)
        np.testing.assert_array_equal(a.fid, b.fid)
        np.testing.assert_array_equal(a.sex, b.sex)
        assert a.generation == b.generation
        assert list(a.extra) == list(b.extra)
        for k in a.extra:
            np.testing.assert_array_equal(a.extra[k], b.extra[k])

    def test_phenotype_npz_roundtrip(self, tmp_path):
        s = self._rich_samples()
        path = str(tmp_path / 'p.npz')
        xio.save_phenotypes_npz(PhenotypeArray(s, {'Y': np.ones(s.n)}), path)
        back = xio.load_phenotypes_npz(path).samples
        self._assert_same(s, back)
        assert back.extra['population'].dtype.kind == 'U'
        assert back.extra['deme'].dtype == np.int32
        # Loadable without pickle
        np.load(path, allow_pickle=False)['sample_extra__population']

    def test_haplotype_npz_roundtrip(self, tmp_path):
        s = self._rich_samples()
        path = str(tmp_path / 'h.npz')
        xio.save_haplotypes_npz(DenseHaplotypeArray(
            genotypes=np.zeros((s.n, 3, 2), dtype=np.int8), generation=4,
            samples=s), path)
        self._assert_same(s, xio.load_haplotypes_npz(path).samples)

    def test_no_extra_roundtrip(self, tmp_path):
        s = SampleMeta(iid=np.arange(3))
        path = str(tmp_path / 'p.npz')
        xio.save_phenotypes_npz(PhenotypeArray(s, {}), path)
        assert xio.load_phenotypes_npz(path).samples.extra == {}

    def test_old_files_without_extra_load(self, tmp_path):
        path = str(tmp_path / 'old.npz')
        np.savez(path, sample_iid=np.arange(3), sample_fid=np.arange(3),
                 sample_sex=np.array([0, 1, 0]), pheno_keys=np.array(['Y']),
                 pheno_Y=np.zeros(3))
        back = xio.load_phenotypes_npz(path)
        assert back.samples.extra == {}
        assert back.samples.generation == 0

    def test_mixed_object_field_rejected(self):
        s = SampleMeta(iid=np.arange(2),
                       extra={'bad': np.array([{'a': 1}, 'x'], dtype=object)})
        with pytest.raises(TypeError, match="cannot be saved"):
            xio._samples_to_npz(s)
