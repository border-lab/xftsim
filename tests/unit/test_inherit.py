"""Unit tests for metadata inheritance rules (xftsim.inherit) and SampleMeta helpers."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta
from xftsim.inherit import (
    InheritanceRule, Midparent, Maternal, Paternal, RandomParent, Custom,
    AncestryProportions, UnrestoredCustom, initialize_metadata,
    inherit_metadata, output_keys, handled_keys, check_rules, rule_from_config,
)


def _parents():
    """Four parents: two of each population, with a numeric score."""
    return SampleMeta(
        iid=np.arange(4), sex=np.array([0, 1, 0, 1]),
        extra={
            'population': np.array(['YRI', 'CEU', 'CEU', 'YRI'], dtype=object),
            'score': np.array([1.0, 3.0, 5.0, 7.0]),
            'deme': np.array([10, 20, 30, 40]),
        },
    )


MAT = np.array([0, 0, 2, 2])
PAT = np.array([1, 3, 1, 3])


class TestSampleMetaHelpers:
    def test_with_extra_replace(self):
        s = _parents().with_extra({'x': np.zeros(4)})
        assert list(s.extra) == ['x']

    def test_with_extra_merge_keeps_and_overwrites(self):
        s = _parents().with_extra({'score': np.zeros(4), 'x': np.ones(4)},
                                  replace=False)
        assert set(s.extra) == {'population', 'score', 'deme', 'x'}
        np.testing.assert_array_equal(s.extra['score'], 0.0)

    def test_with_extra_keeps_core_fields(self):
        p = SampleMeta(iid=np.arange(4), fid=np.array([0, 0, 1, 1]), generation=3)
        s = p.with_extra({'x': np.ones(4)})
        np.testing.assert_array_equal(s.fid, p.fid)
        np.testing.assert_array_equal(s.sex, p.sex)
        assert s.generation == 3

    def test_with_extra_validates_length(self):
        with pytest.raises(ValueError, match="length"):
            _parents().with_extra({'x': np.ones(3)})

    def test_to_dataframe(self):
        df = _parents().with_extra({'pc': np.ones((4, 2))}, replace=False).to_dataframe()
        assert list(df.columns) == ['iid', 'fid', 'sex', 'generation',
                                    'population', 'score', 'deme', 'pc_0', 'pc_1']
        assert len(df) == 4
        assert (df['generation'] == 0).all()


class TestSimpleRules:
    def test_midparent(self):
        out = Midparent('score').inherit(_parents(), MAT, PAT, None)
        np.testing.assert_array_equal(out['score'], [2.0, 4.0, 4.0, 6.0])

    def test_midparent_rejects_strings(self):
        with pytest.raises(TypeError, match="non-numeric"):
            Midparent('population').initialize(_parents())

    def test_midparent_requires_key(self):
        with pytest.raises(KeyError, match="missing_field|needs metadata"):
            Midparent('missing_field').initialize(_parents())

    def test_maternal_and_paternal(self):
        p = _parents()
        np.testing.assert_array_equal(
            Maternal('deme').inherit(p, MAT, PAT, None)['deme'], [10, 10, 30, 30])
        np.testing.assert_array_equal(
            Paternal('population').inherit(p, MAT, PAT, None)['population'],
            ['CEU', 'YRI', 'CEU', 'YRI'])

    def test_maternal_preserves_dtype(self):
        out = Maternal('population').inherit(_parents(), MAT, PAT, None)
        assert out['population'].dtype == object

    def test_random_parent_picks_a_parent_consistently_across_keys(self):
        p = _parents()
        n_off = 1000
        mat = np.zeros(n_off, dtype=int)
        pat = np.ones(n_off, dtype=int)
        out = RandomParent('score', 'deme').inherit(
            p, mat, pat, np.random.RandomState(0))
        from_father = out['score'] == 3.0
        assert set(np.unique(out['score'])) == {1.0, 3.0}
        # The same parent is used for both keys of one offspring.
        np.testing.assert_array_equal(out['deme'][from_father], 20)
        np.testing.assert_array_equal(out['deme'][~from_father], 10)
        assert 0.4 < from_father.mean() < 0.6

    def test_deterministic_rules_do_not_touch_rng(self):
        rng = np.random.RandomState(1)
        rng.rand(5)  # advance past the initial key regeneration
        before = rng.get_state()
        for rule in (Midparent('score'), Maternal('deme'), Paternal('deme'),
                     AncestryProportions()):
            parents = _parents()
            parents = parents.with_extra(rule.initialize(parents), replace=False)
            rule.inherit(parents, MAT, PAT, rng)
        after = rng.get_state()
        np.testing.assert_array_equal(after[1], before[1])
        assert after[2:] == before[2:]

    @pytest.mark.parametrize("cls", [Midparent, Maternal, Paternal, RandomParent])
    def test_needs_a_key(self, cls):
        with pytest.raises(ValueError):
            cls()


class TestCustom:
    def test_custom(self):
        rule = Custom('max_score', lambda m, f, rng: {
            'max_score': np.maximum(m['score'], f['score'])})
        out = rule.inherit(_parents(), MAT, PAT, None)
        np.testing.assert_array_equal(out['max_score'], [3.0, 7.0, 5.0, 7.0])

    def test_custom_wrong_keys(self):
        rule = Custom('a', lambda m, f, rng: {'b': np.zeros(4)})
        with pytest.raises(ValueError, match="returned keys"):
            rule.inherit(_parents(), MAT, PAT, None)

    def test_custom_wrong_length(self):
        rule = Custom('a', lambda m, f, rng: {'a': np.zeros(3)})
        with pytest.raises(ValueError, match="shape"):
            rule.inherit(_parents(), MAT, PAT, None)


class TestAncestryProportions:
    def test_one_hot_at_gen0(self):
        new = AncestryProportions().initialize(_parents())
        assert set(new) == {'ancestry_CEU', 'ancestry_YRI'}
        np.testing.assert_array_equal(new['ancestry_YRI'], [1, 0, 0, 1])
        np.testing.assert_array_equal(new['ancestry_CEU'], [0, 1, 1, 0])

    def test_midparent_after_gen0_sums_to_one(self):
        rule = AncestryProportions()
        p = _parents()
        p = p.with_extra(rule.initialize(p), replace=False)
        out = rule.inherit(p, MAT, PAT, None)
        np.testing.assert_array_equal(out['ancestry_YRI'], [0.5, 1.0, 0.0, 0.5])
        np.testing.assert_array_equal(out['ancestry_YRI'] + out['ancestry_CEU'], 1.0)

    def test_stateless_inherit(self):
        """A fresh rule (as after a checkpoint resume) needs no gen-0 state."""
        p = _parents()
        p = p.with_extra(AncestryProportions().initialize(p), replace=False)
        out = AncestryProportions().inherit(p, MAT, PAT, None)
        assert set(out) == {'ancestry_CEU', 'ancestry_YRI'}

    def test_label_sanitizing(self):
        s = SampleMeta(iid=np.arange(2),
                       extra={'population': np.array(['Han Chinese', 'A-B'])})
        assert set(AncestryProportions().initialize(s)) == {
            'ancestry_Han_Chinese', 'ancestry_A_B'}

    def test_label_collision(self):
        s = SampleMeta(iid=np.arange(2),
                       extra={'population': np.array(['A-B', 'A_B'])})
        with pytest.raises(ValueError, match="both map"):
            AncestryProportions().initialize(s)

    def test_invalid_prefix(self):
        with pytest.raises(ValueError, match="identifier"):
            AncestryProportions(prefix='1anc_').initialize(_parents())

    def test_existing_columns_without_source(self):
        """Founders may carry fractional ancestry (e.g. admixed real data)."""
        s = SampleMeta(iid=np.arange(2),
                       extra={'ancestry_A': np.array([0.25, 1.0]),
                              'ancestry_B': np.array([0.75, 0.0])})
        rule = AncestryProportions()
        assert rule.initialize(s) == {}
        assert rule.output_keys(s) == {'ancestry_A', 'ancestry_B'}

    def test_missing_source_and_columns(self):
        with pytest.raises(KeyError, match="population"):
            AncestryProportions().initialize(SampleMeta(iid=np.arange(2)))

    def test_conflicting_existing_column(self):
        p = _parents().with_extra({'ancestry_YRI': np.zeros(4)}, replace=False)
        with pytest.raises(ValueError, match="different"):
            AncestryProportions().initialize(p)

    def test_foreign_prefixed_field_rejected(self):
        """A non-ancestry field with the prefix would be averaged as ancestry."""
        p = _parents().with_extra({'ancestry_pc1': np.arange(4.0)}, replace=False)
        with pytest.raises(ValueError, match="ancestry_pc1"):
            AncestryProportions().initialize(p)

    def test_existing_columns_must_sum_to_one(self):
        s = SampleMeta(iid=np.arange(2),
                       extra={'ancestry_A': np.array([0.5, 1.0]),
                              'ancestry_pc1': np.array([3.0, -1.0])})
        with pytest.raises(ValueError, match="sum to 1"):
            AncestryProportions().initialize(s)

    def test_inherit_without_columns_explains_resume(self):
        with pytest.raises(KeyError, match="checkpoint"):
            AncestryProportions().inherit(_parents(), MAT, PAT, None)

    def test_handles_source_key(self):
        p = _parents()
        p = p.with_extra(AncestryProportions().initialize(p), replace=False)
        assert 'population' in AncestryProportions().handled_keys(p)


class TestRuleSets:
    def test_initialize_and_inherit(self):
        rules = [AncestryProportions(), Maternal('population'), Midparent('score')]
        p = _parents()
        p = p.with_extra(initialize_metadata(rules, p), replace=False)
        out = inherit_metadata(rules, p, MAT, PAT, None)
        assert set(out) == {'ancestry_CEU', 'ancestry_YRI', 'population', 'score'}
        assert output_keys(rules, p) == set(out)
        assert handled_keys(rules, p) == set(out)

    def test_conflicting_rules(self):
        with pytest.raises(ValueError, match="written by both"):
            check_rules([Maternal('deme'), Paternal('deme')], _parents())

    @pytest.mark.parametrize("rule", [
        Midparent('a', 'b'), Maternal('a'), Paternal('a'), RandomParent('a'),
        AncestryProportions(source='pop', prefix='anc_'),
    ])
    def test_config_roundtrip(self, rule):
        back = rule_from_config(rule.to_config())
        assert type(back) is type(rule)
        assert back.to_config() == rule.to_config()

    def test_custom_config_loads_as_placeholder(self):
        rule = rule_from_config(Custom('x', lambda m, f, r: {}).to_config())
        assert isinstance(rule, UnrestoredCustom)
        assert rule.keys == ('x',)
        with pytest.raises(ValueError, match="not saved"):
            rule.inherit(_parents(), MAT, PAT, None)

    def test_unknown_config(self):
        with pytest.raises(ValueError, match="Unknown"):
            rule_from_config({'type': 'Nope'})

    def test_subclass_contract(self):
        class Constant(InheritanceRule):
            keys = ('c',)

            def inherit(self, parents, maternal_idx, paternal_idx, rng):
                return {'c': np.ones(len(maternal_idx))}

        p = _parents().with_extra({'c': np.zeros(4)}, replace=False)
        out = inherit_metadata([Constant()], p, MAT, PAT, None)
        np.testing.assert_array_equal(out['c'], 1.0)
