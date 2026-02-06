"""
pytest configuration and fixtures for xftsim tests.
"""
import warnings
import pytest


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known deprecation warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        yield


@pytest.fixture(scope="session")
def xft():
    """Import and return xftsim module."""
    import xftsim as xft
    return xft


@pytest.fixture
def small_founder_haplotypes(xft):
    """Create small founder haplotypes for fast tests."""
    return xft.founders.founder_haplotypes_uniform_AFs(n=100, m=50)


@pytest.fixture
def medium_founder_haplotypes(xft):
    """Create medium-sized founder haplotypes for tests."""
    return xft.founders.founder_haplotypes_uniform_AFs(n=500, m=200)


# ── New (Phase 1) fixtures ──────────────────────────────────────────────────

@pytest.fixture
def test_genomes():
    """Simple DenseHaplotypeArray for unit tests."""
    from tests.testdata import TestGenomes
    return TestGenomes.simple(n=100, m=50, seed=42)


@pytest.fixture
def test_effects():
    """Simple AdditiveEffects for unit tests."""
    from tests.testdata import TestEffects
    return TestEffects.additive(m=50, h2=0.5, seed=42)


@pytest.fixture
def test_samples():
    """SampleMeta for unit tests."""
    from tests.testdata import TestMeta
    return TestMeta.samples(n=100, n_fam=20, seed=42)


@pytest.fixture
def test_variants():
    """VariantMeta for unit tests."""
    from tests.testdata import TestMeta
    return TestMeta.variants(m=50, n_chrom=2)
