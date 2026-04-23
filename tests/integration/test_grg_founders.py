"""Integration tests: Generate data with MSPrime and load it into a GRG operator"""
import xftsim as xft
from xftsim.founders import founder_haplotypes_from_msprime_grg


def test_msprime_founder_generation():
    print("Starting Test: msprime founder generation to GRG...")

    n_individuals = 10
    seq_len = 100_000  # 100kb

    operator = founder_haplotypes_from_msprime_grg(
        n=n_individuals,
        sequence_length=seq_len,
        mutation_rate=1e-7,
        recombination_rate=1e-8,
    )

    print("Checking results...")

    assert operator.samples.n == n_individuals, f"Expected {n_individuals} samples, got {operator.samples.n}"
    assert len(operator.samples.iid) == n_individuals

    num_variants = operator.variants.m
    print(f"Generated {num_variants} variants.")
    assert num_variants > 0, "Simulation generated 0 variants. Check mutation rate."
    assert len(operator.variants.pos_bp) == num_variants
    assert len(operator.variants.pos_cM) == num_variants

    assert operator._grg.num_samples == n_individuals * 2
    assert operator._grg.num_mutations == num_variants

    print("SUCCESS: MSPrime GRG founder generation works correctly!")


if __name__ == "__main__":
    test_msprime_founder_generation()
