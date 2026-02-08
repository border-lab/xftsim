"""
Tests for xftsim demo simulations.

These tests verify that all built-in demos run successfully and produce
expected outputs. They serve as smoke tests to catch regressions.
"""
import pytest


class TestDemoSimulation:
    """Tests for DemoSimulation class."""

    def test_bgrm_demo_runs(self, xft):
        """Test BGRM (Bivariate GCTA Random Mating) demo runs without error."""
        demo = xft.sim.DemoSimulation('BGRM')
        assert demo is not None
        assert demo.routine == 'BGRM'

    def test_ugrm_demo_runs(self, xft):
        """Test UGRM (Univariate GCTA Random Mating) demo runs without error."""
        demo = xft.sim.DemoSimulation('UGRM')
        assert demo is not None
        assert demo.routine == 'UGRM'

    def test_bgrm_demo_has_results(self, xft):
        """Test BGRM demo produces results."""
        demo = xft.sim.DemoSimulation('BGRM')
        assert demo.results is not None
        assert len(demo.results) > 0

    def test_ugrm_demo_has_results(self, xft):
        """Test UGRM demo produces results."""
        demo = xft.sim.DemoSimulation('UGRM')
        assert demo.results is not None
        assert len(demo.results) > 0

    def test_bgrm_demo_phenotypes(self, xft):
        """Test BGRM demo has correct phenotype names."""
        demo = xft.sim.DemoSimulation('BGRM')
        assert demo.phenotypes is not None
        keys = list(demo.phenotypes.keys)
        key_str = ' '.join(keys).lower()
        assert 'height' in key_str
        assert 'bmd' in key_str

    def test_ugrm_demo_phenotypes(self, xft):
        """Test UGRM demo has correct phenotype names."""
        demo = xft.sim.DemoSimulation('UGRM')
        assert demo.phenotypes is not None
        keys = list(demo.phenotypes.keys)
        key_str = ' '.join(keys).lower()
        assert 'height' in key_str

    def test_demo_custom_parameters(self, xft):
        """Test demo can be initialized with custom n and m parameters."""
        n, m = 500, 100
        demo = xft.sim.DemoSimulation('BGRM', n=n, m=m)
        assert demo._n == n
        assert demo._m == m

    def test_invalid_routine_raises(self, xft):
        """Test that invalid routine name raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            xft.sim.DemoSimulation('INVALID_ROUTINE')

    def test_demo_repr(self, xft):
        """Test demo has informative string representation."""
        demo = xft.sim.DemoSimulation('BGRM')
        repr_str = repr(demo)
        assert 'DemoSimulation' in repr_str
        assert 'BGRM' in repr_str or 'Bivariate' in repr_str

    def test_demo_can_continue_running(self, xft):
        """Test that demo simulation can continue running additional generations."""
        demo = xft.sim.DemoSimulation('BGRM', n=200, m=50)
        initial_generation = demo.generation
        demo.run(1)
        assert demo.generation == initial_generation + 1


class TestAllDemoRoutines:
    """Parametrized tests that run all available demo routines."""

    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_runs(self, xft, routine):
        """Test each demo routine runs successfully."""
        demo = xft.sim.DemoSimulation(routine)
        assert demo.routine == routine
        assert demo.results is not None

    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_has_haplotypes(self, xft, routine):
        """Test each demo routine has haplotype data."""
        demo = xft.sim.DemoSimulation(routine, n=200, m=50)
        assert demo.haplotypes is not None

    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_has_phenotypes(self, xft, routine):
        """Test each demo routine has phenotype data."""
        demo = xft.sim.DemoSimulation(routine, n=200, m=50)
        assert demo.phenotypes is not None
