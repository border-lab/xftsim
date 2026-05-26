"""Integration tests: msprime coalescent simulation into a GRG-backed founder operator."""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")

from xftsim.founders import founder_haplotypes_from_msprime_grg
from xftsim.struct import GraphHaplotypeOperator


N_INDIVIDUALS = 10
SEQ_LEN = 100_000
MUTATION_RATE = 1e-7
RECOMBINATION_RATE = 1e-8


@pytest.fixture(scope="module")
def msprime_operator():
    """Build a GRG founder operator once and share across tests in this module."""
    return founder_haplotypes_from_msprime_grg(
        n=N_INDIVIDUALS,
        sequence_length=SEQ_LEN,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
    )


class TestMsprimeGRGFounders:
    def test_returns_graph_operator(self, msprime_operator):
        """Founder generation should return a GraphHaplotypeOperator."""
        assert isinstance(msprime_operator, GraphHaplotypeOperator)

    def test_sample_count(self, msprime_operator):
        """SampleMeta should reflect the requested number of diploid individuals."""
        assert msprime_operator.samples.n == N_INDIVIDUALS
        assert len(msprime_operator.samples.iid) == N_INDIVIDUALS

    def test_grg_internal_samples(self, msprime_operator):
        """GRG stores two haploid samples per diploid individual."""
        assert msprime_operator._grg.num_samples == 2 * N_INDIVIDUALS

    def test_variant_count_positive(self, msprime_operator):
        """The requested mutation rate should produce at least one variant."""
        assert msprime_operator.variants.m > 0

    def test_variant_metadata_lengths(self, msprime_operator):
        """GRG mutation count and variant arrays should all agree on m."""
        m = msprime_operator.variants.m
        assert msprime_operator._grg.num_mutations == m
        assert len(msprime_operator.variants.pos_bp) == m
        assert len(msprime_operator.variants.pos_cM) == m

    def test_default_chrom_label(self, msprime_operator):
        """msprime path assigns the placeholder chromosome label '1' to every variant."""
        assert np.all(msprime_operator.variants.chrom == "1")

    def test_pos_cM_scaling(self, msprime_operator):
        """pos_cM should be pos_bp * recombination_rate * 100."""
        expected = msprime_operator.variants.pos_bp * RECOMBINATION_RATE * 100.0
        np.testing.assert_allclose(msprime_operator.variants.pos_cM, expected)
