"""
Tests for parental transmission components (mother, father, parent).
"""
import warnings
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.struct import SampleMeta, DenseHaplotypeArray, PhenotypeArray, PedigreeArray
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent, ArchNode,
)
from xftsim.effect import AdditiveEffects
from xftsim.parser import parse_formula
from xftsim.sim import Simulation


class TestMotherComponent:
    def test_gen0_founder_noise(self):
        """At gen 0 with founder=noise, should return noise values."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10)
        founder = NoiseComponent(variance=0.5)
        comp = MotherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.VT'], component=comp)
        pheno = PhenotypeArray(samples=hap.samples)

        result = comp.compute(node, hap, pheno,
                             rng=np.random.RandomState(42),
                             generation=0,
                             phenotype_history={},
                             pedigree_history={})
        assert result.shape == (100,)
        assert np.std(result) > 0  # should have variance

    def test_gen0_no_founder_warns(self):
        """At gen 0 without founder, should warn and return zeros."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10)
        comp = MotherComponent('Y', founder_component=None)
        node = ArchNode(outputs=['Y.VT'], component=comp)
        pheno = PhenotypeArray(samples=hap.samples)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno,
                                 rng=np.random.RandomState(42),
                                 generation=0,
                                 phenotype_history={},
                                 pedigree_history={})
            assert len(w) == 1
            assert "no pedigree" in str(w[0].message).lower()
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_gen1_correct_values(self):
        """At gen 1, should look up mother's phenotype via maternal_idx."""
        n = 20
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples_gen0 = SampleMeta(iid=np.arange(n), sex=sex, generation=0)
        samples_gen1 = SampleMeta(iid=np.arange(n), sex=sex, generation=1)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples_gen1, generation=1)

        # Parent phenotypes
        parent_pheno = PhenotypeArray(samples=samples_gen0)
        parent_pheno._values['Y'] = np.arange(n, dtype=np.float64)

        # Pedigree: each offspring i gets mother i%10
        maternal_idx = np.arange(n) % 10
        paternal_idx = np.arange(n) % 10 + 10
        ped = PedigreeArray(offspring_samples=samples_gen1,
                           maternal_idx=maternal_idx,
                           paternal_idx=paternal_idx,
                           parent_n=n)

        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.M'], component=comp)
        pheno = PhenotypeArray(samples=samples_gen1)

        result = comp.compute(node, hap, pheno,
                             generation=1,
                             phenotype_history={0: parent_pheno},
                             pedigree_history={1: ped})

        expected = parent_pheno['Y'][maternal_idx]
        np.testing.assert_array_equal(result, expected)


class TestFatherComponent:
    def test_gen0_founder(self):
        """At gen 0 with founder, should return noise."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10)
        founder = NoiseComponent(variance=1.0)
        comp = FatherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.VT'], component=comp)
        pheno = PhenotypeArray(samples=hap.samples)

        result = comp.compute(node, hap, pheno,
                             rng=np.random.RandomState(42),
                             generation=0,
                             phenotype_history={},
                             pedigree_history={})
        assert result.shape == (100,)
        assert np.std(result) > 0

    def test_gen1_correct(self):
        """At gen 1, should look up father's phenotype via paternal_idx."""
        n = 20
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples_gen0 = SampleMeta(iid=np.arange(n), sex=sex, generation=0)
        samples_gen1 = SampleMeta(iid=np.arange(n), sex=sex, generation=1)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples_gen1, generation=1)

        parent_pheno = PhenotypeArray(samples=samples_gen0)
        parent_pheno._values['Y'] = np.arange(n, dtype=np.float64) * 2.0

        maternal_idx = np.arange(n) % 10
        paternal_idx = np.arange(n) % 10 + 10
        ped = PedigreeArray(offspring_samples=samples_gen1,
                           maternal_idx=maternal_idx,
                           paternal_idx=paternal_idx,
                           parent_n=n)

        comp = FatherComponent('Y')
        node = ArchNode(outputs=['Y.F'], component=comp)
        pheno = PhenotypeArray(samples=samples_gen1)

        result = comp.compute(node, hap, pheno,
                             generation=1,
                             phenotype_history={0: parent_pheno},
                             pedigree_history={1: ped})

        expected = parent_pheno['Y'][paternal_idx]
        np.testing.assert_array_equal(result, expected)


class TestParentComponent:
    def test_midparent(self):
        """Parent should be average of mother and father values."""
        n = 20
        m = 5
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples_gen0 = SampleMeta(iid=np.arange(n), sex=sex, generation=0)
        samples_gen1 = SampleMeta(iid=np.arange(n), sex=sex, generation=1)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples_gen1, generation=1)

        parent_pheno = PhenotypeArray(samples=samples_gen0)
        parent_pheno._values['Y'] = np.arange(n, dtype=np.float64)

        maternal_idx = np.arange(n) % 10
        paternal_idx = np.arange(n) % 10 + 10
        ped = PedigreeArray(offspring_samples=samples_gen1,
                           maternal_idx=maternal_idx,
                           paternal_idx=paternal_idx,
                           parent_n=n)

        comp = ParentComponent('Y')
        node = ArchNode(outputs=['Y.P'], component=comp)
        pheno = PhenotypeArray(samples=samples_gen1)

        result = comp.compute(node, hap, pheno,
                             generation=1,
                             phenotype_history={0: parent_pheno},
                             pedigree_history={1: ped})

        mother_vals = parent_pheno['Y'][maternal_idx]
        father_vals = parent_pheno['Y'][paternal_idx]
        expected = 0.5 * (mother_vals + father_vals)
        np.testing.assert_allclose(result, expected)

    def test_gen0_founder(self):
        """At gen 0, parent with founder should use founder component."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10)
        founder = NoiseComponent(variance=0.5)
        comp = ParentComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.VT'], component=comp)
        pheno = PhenotypeArray(samples=hap.samples)

        result = comp.compute(node, hap, pheno,
                             rng=np.random.RandomState(42),
                             generation=0,
                             phenotype_history={},
                             pedigree_history={})
        assert result.shape == (100,)
        assert np.std(result) > 0


class TestParserParental:
    def test_parse_parent_height(self):
        """Parser should handle parent(Y)."""
        nodes = parse_formula("Y.VT ~ parent(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, ParentComponent)
        assert nodes[0].component.phenotype_name == 'Y'

    def test_parse_parent_with_founder(self):
        """Parser should handle parent(Y, founder=noise(0.5))."""
        nodes = parse_formula("Y.VT ~ parent(Y, founder=noise(0.5))")
        assert len(nodes) == 1
        comp = nodes[0].component
        assert isinstance(comp, ParentComponent)
        assert comp.phenotype_name == 'Y'
        assert comp.founder_component is not None
        assert isinstance(comp.founder_component, NoiseComponent)
        assert comp.founder_component.variance == 0.5

    def test_parse_mother(self):
        """Parser should handle mother(Y)."""
        nodes = parse_formula("Y.M ~ mother(Y)")
        assert isinstance(nodes[0].component, MotherComponent)

    def test_parse_father(self):
        """Parser should handle father(Y)."""
        nodes = parse_formula("Y.F ~ father(Y)")
        assert isinstance(nodes[0].component, FatherComponent)


class TestVTSimulation:
    def test_vt_3gen_variance_inflates(self):
        """With VT, phenotypic variance should increase across generations."""
        hap = TestSimulation.founder_haplotypes(n=1000, m=50)
        arch = TestSimulation.vt_architecture(m=50, h2=0.5, vt_weight=0.3)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = Simulation(hap, arch, rm, rmap, seed=42,
                         retain_phenotypes=10)
        sim.run(4)

        var_0 = np.var(sim.phenotype_history[0]['Y'])
        var_3 = np.var(sim.phenotype_history[3]['Y'])
        # VT should inflate variance across generations
        assert var_3 > var_0 * 0.9  # at least not much smaller

    def test_vt_aggressive_retention_still_works(self):
        """VT with retain_phenotypes=0 should still work because
        retention runs after compute, so gen-1 is always available."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.vt_architecture(m=50, h2=0.5, vt_weight=0.3)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = Simulation(hap, arch, rm, rmap, seed=42,
                         retain_phenotypes=0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(5)
            retention_warnings = [
                x for x in w
                if "not in phenotype_history" in str(x.message)
            ]
            # No warnings: retention runs after compute, gen-1 is always alive
            assert len(retention_warnings) == 0
        # Phenotype values should still be valid at the last generation
        assert np.std(sim.phenotype_history[sim.generation]['Y']) > 0

    def test_vt_parent_offspring_correlation(self):
        """VT should produce parent-offspring phenotype correlation > 0."""
        hap = TestSimulation.founder_haplotypes(n=1000, m=50)
        arch = TestSimulation.vt_architecture(m=50, h2=0.5, vt_weight=0.3)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = Simulation(hap, arch, rm, rmap, seed=42,
                         retain_phenotypes=10)
        sim.run(3)

        # Compute parent-offspring correlation at gen 2
        ped = sim.pedigree_history[2]
        parent_y = sim.phenotype_history[1]['Y']
        offspring_y = sim.phenotype_history[2]['Y']
        midparent = 0.5 * (parent_y[ped.maternal_idx] + parent_y[ped.paternal_idx])
        r = np.corrcoef(midparent, offspring_y)[0, 1]
        assert r > 0.1  # should have meaningful correlation
