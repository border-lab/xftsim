"""
Unit tests for mating regime serialization/deserialization.

Tests:
1. RandomMating serialization roundtrip
2. LinearAssortativeMating serialization roundtrip
3. Unknown mating type deserialization raises ValueError
4. Assortative mating preserves all parameters
5. Custom mating type serialization (type name only)
"""
import numpy as np
import pytest

from xftsim.io import _serialize_mating_regime, _deserialize_mating_regime
from xftsim.mate import RandomMating, LinearAssortativeMating


class TestRandomMatingRoundtrip:
    def test_roundtrip_default(self):
        """RandomMating with default offspring_per_pair."""
        regime = RandomMating(offspring_per_pair=2)
        data = _serialize_mating_regime(regime)
        restored = _deserialize_mating_regime(data)
        assert isinstance(restored, RandomMating)
        assert restored.offspring_per_pair == 2

    def test_roundtrip_custom_opp(self):
        """RandomMating with custom offspring_per_pair."""
        regime = RandomMating(offspring_per_pair=5)
        data = _serialize_mating_regime(regime)
        restored = _deserialize_mating_regime(data)
        assert restored.offspring_per_pair == 5

    def test_serialization_format(self):
        regime = RandomMating(offspring_per_pair=3)
        data = _serialize_mating_regime(regime)
        assert data['type'] == 'RandomMating'
        assert data['offspring_per_pair'] == 3


class TestAssortativeMatingRoundtrip:
    def test_roundtrip(self):
        regime = LinearAssortativeMating(
            component_names=['Y', 'Z'],
            r=0.5,
            offspring_per_pair=3,
        )
        data = _serialize_mating_regime(regime)
        restored = _deserialize_mating_regime(data)
        assert isinstance(restored, LinearAssortativeMating)
        assert restored.component_names == ['Y', 'Z']
        assert restored.r == 0.5
        assert restored.offspring_per_pair == 3

    def test_negative_r_preserved(self):
        regime = LinearAssortativeMating(
            component_names=['Y'],
            r=-0.3,
        )
        data = _serialize_mating_regime(regime)
        restored = _deserialize_mating_regime(data)
        assert restored.r == -0.3

    def test_serialization_format(self):
        regime = LinearAssortativeMating(
            component_names=['Y'],
            r=0.5,
            offspring_per_pair=2,
        )
        data = _serialize_mating_regime(regime)
        assert data['type'] == 'LinearAssortativeMating'
        assert data['component_names'] == ['Y']
        assert data['r'] == 0.5
        assert data['offspring_per_pair'] == 2


class TestUnknownMatingType:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown mating regime"):
            _deserialize_mating_regime({'type': 'CustomMatingRegime'})

    def test_custom_class_raises_on_serialize(self):
        """Unsupported regimes fail loud at save time rather than silently
        producing a stub config that only fails on load. Previously this
        returned ``{'type': 'CustomMating'}`` and dropped all params.
        """
        class CustomMating:
            pass
        with pytest.raises(ValueError, match="[Cc]annot serialize"):
            _serialize_mating_regime(CustomMating())
