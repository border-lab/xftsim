"""
Unit tests for ArchComponent kind/type attributes.

Tests:
1. GeneticComponent kind = 'genetic'
2. NoiseComponent kind = 'generative'
3. AggregationComponent kind = 'aggregating'
4. CNoiseComponent kind = 'generative'
5. MotherComponent kind = 'reference'
6. SiblingMeanComponent kind = 'aggregating'
7. Each component accepts_grouping properly set
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.narch import (
    GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent,
    AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingSumComponent,
    SiblingCountComponent, SiblingAnyComponent,
    SiblingEldestComponent, SiblingYoungestComponent,
)


class TestComponentKind:
    def test_genetic_kind(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        comp = GeneticComponent(eff)
        assert comp.kind == 'genetic'

    def test_mvgenetic_kind(self):
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        comp = MVGeneticComponent(mv)
        assert comp.kind == 'genetic'

    def test_haplotype_genetic_kind(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        comp = HaplotypeGeneticComponent(eff, haplotype='maternal')
        assert comp.kind == 'genetic'

    def test_noise_kind(self):
        comp = NoiseComponent(variance=1.0)
        assert comp.kind == 'generative'

    def test_cnoise_kind(self):
        comp = CNoiseComponent(cov=np.eye(2))
        assert comp.kind == 'generative'

    def test_aggregation_kind(self):
        comp = AggregationComponent('A + B')
        assert comp.kind == 'aggregating'

    def test_mother_kind(self):
        comp = MotherComponent('Y')
        assert comp.kind == 'reference'

    def test_father_kind(self):
        comp = FatherComponent('Y')
        assert comp.kind == 'reference'

    def test_parent_kind(self):
        comp = ParentComponent('Y')
        assert comp.kind == 'reference'

    def test_sibling_mean_kind(self):
        comp = SiblingMeanComponent('Y')
        assert comp.kind == 'aggregating'

    def test_sibling_sum_kind(self):
        comp = SiblingSumComponent('Y')
        assert comp.kind == 'aggregating'

    def test_sibling_count_kind(self):
        comp = SiblingCountComponent('Y')
        assert comp.kind == 'aggregating'


class TestAcceptsGrouping:
    def test_noise_accepts_grouping(self):
        comp = NoiseComponent(variance=1.0)
        assert comp.accepts_grouping is True

    def test_cnoise_accepts_grouping(self):
        comp = CNoiseComponent(cov=np.eye(2))
        assert comp.accepts_grouping is True

    def test_genetic_no_grouping(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        comp = GeneticComponent(eff)
        assert comp.accepts_grouping is False

    def test_aggregation_no_grouping(self):
        comp = AggregationComponent('A + B')
        assert comp.accepts_grouping is False

    def test_sibling_mean_accepts_grouping(self):
        comp = SiblingMeanComponent('Y')
        assert comp.accepts_grouping is True

    def test_sibling_any_accepts_grouping(self):
        comp = SiblingAnyComponent('Y')
        assert comp.accepts_grouping is True
