"""Integration tests: stdpopsim demographic-model simulation into a GRG-backed founder operator."""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")
pytest.importorskip("stdpopsim")

from xftsim.founders import founder_haplotypes_from_stdpopsim_grg
from xftsim.struct import GraphHaplotypeOperator


SAMPLES = {"YRI": 5, "CEU": 5, "CHB": 5}
N_INDIVIDUALS = sum(SAMPLES.values())
MODEL_ID = "OutOfAfrica_3G09"
CHROMOSOME = "chr22"
SPECIES_ID = "HomSap"
LEFT = 0
RIGHT = 500_000


@pytest.fixture(scope="module")
def stdpopsim_operator():
    """Build a GRG founder operator from stdpopsim once and share across tests in this module."""
    return founder_haplotypes_from_stdpopsim_grg(
        samples=SAMPLES,
        model_id=MODEL_ID,
        chromosome=CHROMOSOME,
        species_id=SPECIES_ID,
        left=LEFT,
        right=RIGHT,
    )


class TestStdpopsimGRGFounders:
    def test_returns_graph_operator(self, stdpopsim_operator):
        """Founder generation should return a GraphHaplotypeOperator."""
        assert isinstance(stdpopsim_operator, GraphHaplotypeOperator)

    def test_sample_count(self, stdpopsim_operator):
        """SampleMeta.n equals the total of all per-population sample counts."""
        assert stdpopsim_operator.samples.n == N_INDIVIDUALS
        assert len(stdpopsim_operator.samples.iid) == N_INDIVIDUALS

    def test_grg_internal_samples(self, stdpopsim_operator):
        """GRG stores two haploid samples per diploid individual."""
        assert stdpopsim_operator._grg.num_samples == 2 * N_INDIVIDUALS

    def test_variant_count_positive(self, stdpopsim_operator):
        """The requested window should produce at least one variant."""
        assert stdpopsim_operator.variants.m > 0

    def test_variant_metadata_lengths(self, stdpopsim_operator):
        """GRG mutation count and variant arrays should all agree on m."""
        m = stdpopsim_operator.variants.m
        assert stdpopsim_operator._grg.num_mutations == m
        assert len(stdpopsim_operator.variants.pos_bp) == m
        assert len(stdpopsim_operator.variants.pos_cM) == m

    def test_chrom_metadata_matches_request(self, stdpopsim_operator):
        """Every variant should carry the requested chromosome label."""
        assert np.all(stdpopsim_operator.variants.chrom == CHROMOSOME)
