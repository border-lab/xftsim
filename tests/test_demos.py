"""
Tests for xftsim demo simulations.

These tests verify that all built-in demos run successfully and produce
expected outputs. They serve as smoke tests to catch regressions.
"""
import pytest


class TestDemoSimulation:
    """Tests for DemoSimulation class."""

    @pytest.mark.timeout(60)
    def test_bgrm_demo_runs(self, xft):
        """Test BGRM (Bivariate GCTA Random Mating) demo runs without error."""
        demo = xft.sim.DemoSimulation('BGRM')
        assert demo is not None
        assert demo.routine == 'BGRM'

    @pytest.mark.timeout(60)
    def test_ugrm_demo_runs(self, xft):
        """Test UGRM (Univariate GCTA Random Mating) demo runs without error."""
        demo = xft.sim.DemoSimulation('UGRM')
        assert demo is not None
        assert demo.routine == 'UGRM'

    @pytest.mark.timeout(60)
    def test_bgrm_demo_has_results(self, xft):
        """Test BGRM demo produces expected result keys."""
        demo = xft.sim.DemoSimulation('BGRM')

        assert hasattr(demo, 'results')
        assert 'sample_statistics' in demo.results
        assert 'mating_statistics' in demo.results

    @pytest.mark.timeout(60)
    def test_ugrm_demo_has_results(self, xft):
        """Test UGRM demo produces expected result keys."""
        demo = xft.sim.DemoSimulation('UGRM')

        assert hasattr(demo, 'results')
        assert 'sample_statistics' in demo.results
        assert 'mating_statistics' in demo.results

    @pytest.mark.timeout(60)
    def test_bgrm_demo_phenotypes(self, xft):
        """Test BGRM demo has correct phenotype names."""
        demo = xft.sim.DemoSimulation('BGRM')

        # BGRM should have height and BMD phenotypes
        assert hasattr(demo, 'phenotypes')
        phenotype_names = list(demo.phenotypes.coords['phenotype_name'].values)
        assert 'height' in str(phenotype_names).lower()
        assert 'bmd' in str(phenotype_names).lower()

    @pytest.mark.timeout(60)
    def test_ugrm_demo_phenotypes(self, xft):
        """Test UGRM demo has correct phenotype names."""
        demo = xft.sim.DemoSimulation('UGRM')

        # UGRM should have height phenotype
        assert hasattr(demo, 'phenotypes')
        phenotype_names = list(demo.phenotypes.coords['phenotype_name'].values)
        assert 'height' in str(phenotype_names).lower()

    @pytest.mark.timeout(60)
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

    @pytest.mark.timeout(60)
    def test_demo_can_continue_running(self, xft):
        """Test that demo simulation can continue running additional generations."""
        demo = xft.sim.DemoSimulation('BGRM', n=200, m=50)
        initial_generation = demo.generation

        # Run one more generation
        demo.run(1)

        assert demo.generation == initial_generation + 1


class TestAllDemoRoutines:
    """Parametrized tests that run all available demo routines."""

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_runs(self, xft, routine):
        """Test each demo routine runs successfully."""
        demo = xft.sim.DemoSimulation(routine)
        assert demo.routine == routine
        assert hasattr(demo, 'results')

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_has_haplotypes(self, xft, routine):
        """Test each demo routine has haplotype data."""
        demo = xft.sim.DemoSimulation(routine, n=200, m=50)
        assert hasattr(demo, 'haplotypes')
        assert demo.haplotypes is not None

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("routine", ["UGRM", "BGRM"])
    def test_demo_routine_has_phenotypes(self, xft, routine):
        """Test each demo routine has phenotype data."""
        demo = xft.sim.DemoSimulation(routine, n=200, m=50)
        assert hasattr(demo, 'phenotypes')
        assert demo.phenotypes is not None
