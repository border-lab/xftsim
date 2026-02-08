"""
Numerical test: founder genotype frequencies follow Hardy-Weinberg equilibrium.

Tests:
1. Founder allele frequencies near target (0.5 ± sampling noise)
2. Genotype frequencies match HWE expectations: p², 2pq, q²
3. Heterozygosity matches expected 2pq
4. No monomorphic loci in large sample
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestFounderHWE:
    def test_allele_frequencies_near_half(self):
        """Founder haplotypes drawn uniformly → AF ≈ 0.5."""
        hap = TestSimulation.founder_haplotypes(n=1000, m=50, seed=42)
        af = hap.recompute_af()
        # With n=1000 haploid samples (2000 alleles), AF should be near 0.5
        assert np.all(af > 0.1), "Some AF too low"
        assert np.all(af < 0.9), "Some AF too high"
        # Mean AF should be close to 0.5
        assert abs(np.mean(af) - 0.5) < 0.05

    def test_genotype_frequencies_hwe(self):
        """Genotype frequencies should match HWE: p², 2pq, q²."""
        hap = TestSimulation.founder_haplotypes(n=2000, m=20, seed=42)
        geno = hap.genotypes  # (n, m, 2)
        dosage = geno[:, :, 0] + geno[:, :, 1]  # 0, 1, or 2

        af = hap.recompute_af()
        for j in range(20):
            p = af[j]
            q = 1 - p
            n = 2000
            # Observed frequencies
            f_aa = np.mean(dosage[:, j] == 0)
            f_Aa = np.mean(dosage[:, j] == 1)
            f_AA = np.mean(dosage[:, j] == 2)
            # Expected HWE frequencies
            exp_aa = q ** 2
            exp_Aa = 2 * p * q
            exp_AA = p ** 2
            # Allow sampling tolerance (chi-square-like, wide tolerance)
            assert abs(f_aa - exp_aa) < 0.1, \
                f"Locus {j}: f(aa)={f_aa:.3f}, expected {exp_aa:.3f}"
            assert abs(f_Aa - exp_Aa) < 0.1, \
                f"Locus {j}: f(Aa)={f_Aa:.3f}, expected {exp_Aa:.3f}"
            assert abs(f_AA - exp_AA) < 0.1, \
                f"Locus {j}: f(AA)={f_AA:.3f}, expected {exp_AA:.3f}"

    def test_heterozygosity_matches_expected(self):
        """Observed heterozygosity ≈ expected 2pq across loci."""
        hap = TestSimulation.founder_haplotypes(n=2000, m=30, seed=123)
        geno = hap.genotypes
        dosage = geno[:, :, 0] + geno[:, :, 1]
        af = hap.recompute_af()

        for j in range(30):
            p = af[j]
            obs_het = np.mean(dosage[:, j] == 1)
            exp_het = 2 * p * (1 - p)
            assert abs(obs_het - exp_het) < 0.08, \
                f"Locus {j}: obs_het={obs_het:.3f}, exp_het={exp_het:.3f}"

    def test_no_monomorphic_loci_large_sample(self):
        """With n=1000, all loci should be polymorphic (AF ≠ 0 or 1)."""
        hap = TestSimulation.founder_haplotypes(n=1000, m=50, seed=42)
        af = hap.recompute_af()
        assert np.all(af > 0.0), "Monomorphic locus (AF=0) found"
        assert np.all(af < 1.0), "Fixed locus (AF=1) found"
