"""Unit tests for sibling indirect genetic effects (exclude_self leave-one-out).

A sibling IGE is the influence of an individual's *siblings'* values on that
individual, so it excludes self. Expected values are computed from the spec,
not the code under test:

    sum,  exclude_self:  out_i = sum_group - v_i
    mean, exclude_self:  out_i = (sum_group - v_i) / (n_group - 1)  (singletons 0)

The default (exclude_self=False) behavior of the existing components is
unchanged; only sibling_sum / sibling_mean support leave-one-out.
"""
import numpy as np
import pytest

from xftsim.arch import (
    SiblingSumComponent, SiblingMeanComponent, SiblingCountComponent,
    SiblingAnyComponent, SiblingEldestComponent, SiblingYoungestComponent,
    Architecture,
)
from xftsim.parser import _parse_sibling
from xftsim.effect import AdditiveEffects
from xftsim.io import save_architecture, load_architecture


# groups: {idx 0,1,2}, {idx 3,4}, {idx 5 singleton}
LABELS = np.array([0, 0, 0, 1, 1, 2])
VALUES = np.array([1., 2., 3., 10., 20., 99.])


class TestLeaveOneOutAggregation:
    def test_sum_exclude_self(self):
        out = SiblingSumComponent('g', exclude_self=True)._aggregate_groups(VALUES, LABELS)
        np.testing.assert_allclose(out, [5, 4, 3, 20, 10, 0])          # sum_group - v_i

    def test_mean_exclude_self(self):
        out = SiblingMeanComponent('g', exclude_self=True)._aggregate_groups(VALUES, LABELS)
        np.testing.assert_allclose(out, [2.5, 2.0, 1.5, 20.0, 10.0, 0.0])  # (sum-v)/(n-1)

    def test_singleton_is_zero(self):
        labels = np.array([0, 1, 2]); values = np.array([5., 6., 7.])
        for cls in (SiblingSumComponent, SiblingMeanComponent):
            out = cls('g', exclude_self=True)._aggregate_groups(values, labels)
            np.testing.assert_array_equal(out, [0, 0, 0])

    def test_self_inclusive_unchanged(self):
        # Regression: default behavior is the original self-inclusive aggregate.
        np.testing.assert_allclose(
            SiblingSumComponent('g')._aggregate_groups(VALUES, LABELS), [6, 6, 6, 30, 30, 99])
        np.testing.assert_allclose(
            SiblingMeanComponent('g')._aggregate_groups(VALUES, LABELS), [2, 2, 2, 15, 15, 99])


class TestExcludeSelfGuard:
    @pytest.mark.parametrize("cls", [SiblingCountComponent, SiblingAnyComponent,
                                     SiblingEldestComponent, SiblingYoungestComponent])
    def test_unsupported_components_reject(self, cls):
        with pytest.raises(ValueError, match="does not support exclude_self"):
            cls('g', exclude_self=True)

    def test_supported_default_false(self):
        assert SiblingSumComponent('g').exclude_self is False
        assert SiblingMeanComponent('g').exclude_self is False

    def test_repr_shows_flag(self):
        assert "exclude_self=True" in repr(SiblingSumComponent('g', exclude_self=True))
        assert "exclude_self" not in repr(SiblingSumComponent('g'))


class TestParser:
    def test_parses_exclude_self_true(self):
        node = _parse_sibling('sibling_sum', ['ige'], 'g, exclude_self=True', 1)
        assert node.component.exclude_self is True
        assert node.inputs == ['g']

    def test_parses_exclude_self_false(self):
        node = _parse_sibling('sibling_mean', ['x'], 'g, exclude_self=False', 1)
        assert node.component.exclude_self is False

    def test_default_when_omitted(self):
        assert _parse_sibling('sibling_sum', ['x'], 'g', 1).component.exclude_self is False

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="unexpected argument"):
            _parse_sibling('sibling_sum', ['x'], 'g, foo=True', 1)

    def test_non_bool_rejected(self):
        with pytest.raises(ValueError, match="must be True or False"):
            _parse_sibling('sibling_sum', ['x'], 'g, exclude_self=maybe', 1)

    def test_parser_wraps_guard_for_unsupported(self):
        with pytest.raises(ValueError, match="does not support exclude_self"):
            _parse_sibling('sibling_count', ['x'], 'g, exclude_self=True', 7)


class TestIORoundtrip:
    @pytest.mark.parametrize("cls", [SiblingSumComponent, SiblingMeanComponent])
    def test_exclude_self_survives_roundtrip(self, cls, tmp_path):
        arch = Architecture()
        arch.add('Y.ige', cls('Y', exclude_self=True), inputs=['Y'], grouping='FID')
        d = str(tmp_path / 'arch')
        save_architecture(arch, d)
        comp = load_architecture(d)._nodes[0].component
        assert isinstance(comp, cls)
        assert comp.exclude_self is True

    def test_default_false_survives_roundtrip(self, tmp_path):
        arch = Architecture()
        arch.add('Y.sib', SiblingSumComponent('Y'), inputs=['Y'], grouping='FID')
        d = str(tmp_path / 'arch')
        save_architecture(arch, d)
        assert load_architecture(d)._nodes[0].component.exclude_self is False
