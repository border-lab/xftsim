"""Integration tests: Generate data with stdpopsim and load it into a GRG operator"""
import xftsim as xft
from xftsim.founders import founder_haplotypes_from_stdpopsim_grg


def test_stdpopsim_founder_generation():
    print("Starting Test: stdpopsim founder generation to GRG...")

    samples = {"YRI": 5, "CEU": 5, "CHB": 5}
    n_individuals = sum(samples.values())

    operator = founder_haplotypes_from_stdpopsim_grg(
        samples=samples,
        model_id="OutOfAfrica_3G09",
        chromosome="chr22",
        species_id="HomSap",
        left=0,
        right=500_000,
    )

    print("Checking results...")

    assert operator.samples.n == n_individuals, f"Expected {n_individuals} samples, got {operator.samples.n}"
    assert len(operator.samples.iid) == n_individuals

    num_variants = operator.variants.m
    print(f"Generated {num_variants} variants.")
    assert num_variants > 0, "Simulation generated 0 variants. Check model/length_multiplier."
    assert len(operator.variants.pos_bp) == num_variants
    assert len(operator.variants.pos_cM) == num_variants

    assert (operator.variants.chrom == "chr22").all(), "Variant chromosome metadata should match requested contig."

    assert operator._grg.num_samples == n_individuals * 2
    assert operator._grg.num_mutations == num_variants

    print("SUCCESS: stdpopsim GRG founder generation works correctly!")


if __name__ == "__main__":
    test_stdpopsim_founder_generation()
