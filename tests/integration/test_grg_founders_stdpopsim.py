"""Integration tests: stdpopsim demographic-model simulation into a GRG-backed founder operator."""
from collections import Counter

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

    def test_iid_prefixed_with_population(self, stdpopsim_operator):
        """Every iid should start with one of the requested population names."""
        pops = set(SAMPLES.keys())
        for iid in stdpopsim_operator.samples.iid:
            prefix = str(iid).split("_", 1)[0]
            assert prefix in pops, f"iid {iid!r} has unknown population prefix"

    def test_population_extra_present(self, stdpopsim_operator):
        """SampleMeta.extra should carry a per-individual population label."""
        extra = stdpopsim_operator.samples.extra
        assert "population" in extra
        assert len(extra["population"]) == N_INDIVIDUALS

    def test_population_counts_match_request(self, stdpopsim_operator):
        """Per-population sample counts in extra should match the input dict."""
        pops = stdpopsim_operator.samples.extra["population"]
        assert Counter(map(str, pops)) == SAMPLES

    def test_pos_bp_monotonic(self, stdpopsim_operator):
        """Variant positions should be non-decreasing (sites sorted by position)."""
        pos_bp = stdpopsim_operator.variants.pos_bp
        assert np.all(np.diff(pos_bp) >= 0)

    def test_pos_bp_in_window(self, stdpopsim_operator):
        """All variant positions should fall inside the requested [LEFT, RIGHT) window."""
        pos_bp = stdpopsim_operator.variants.pos_bp
        assert np.all((pos_bp >= LEFT) & (pos_bp < RIGHT))

    def test_pos_cM_monotonic(self, stdpopsim_operator):
        """Cumulative cM should be non-decreasing."""
        pos_cM = stdpopsim_operator.variants.pos_cM
        assert np.all(np.diff(pos_cM) >= 0)

    def test_alleles_not_placeholder(self, stdpopsim_operator):
        """Ref/alt alleles should be real strings, not the old '0'/'1' placeholders."""
        z = stdpopsim_operator.variants.zero_allele
        o = stdpopsim_operator.variants.one_allele
        assert not np.all(z == "0")
        assert not np.all(o == "1")

    def test_vid_structured(self, stdpopsim_operator):
        """vid format should be '{chrom}:{pos}:{ref}:{alt}'."""
        vid0 = str(stdpopsim_operator.variants.vid[0])
        parts = vid0.split(":")
        assert len(parts) >= 4
        assert parts[0] == CHROMOSOME
