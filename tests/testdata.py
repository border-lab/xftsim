"""
Deterministic test data generators for xftsim tests.

All generators use explicit seeds for reproducibility.
Usable both inside pytest (via conftest fixtures) and standalone.
"""
import numpy as np
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestGenomes:
    """Factory for test haplotype arrays."""

    @staticmethod
    def simple(n=500, m=100, seed=42) -> DenseHaplotypeArray:
        """Simple random haplotypes with uniform AFs."""
        rng = np.random.RandomState(seed)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        return DenseHaplotypeArray(genotypes=geno)

    @staticmethod
    def biallelic_known_af(n, af, seed=42) -> DenseHaplotypeArray:
        """
        Haplotypes with known allele frequencies.

        Parameters
        ----------
        n : int
            Number of samples.
        af : array-like
            Allele frequencies per variant.
        seed : int
            Random seed.
        """
        af = np.asarray(af, dtype=np.float64)
        m = len(af)
        rng = np.random.RandomState(seed)
        geno = np.zeros((n, m, 2), dtype=np.int8)
        for j in range(m):
            geno[:, j, 0] = rng.binomial(1, af[j], size=n)
            geno[:, j, 1] = rng.binomial(1, af[j], size=n)
        variants = VariantMeta(vid=np.arange(m), af=af)
        return DenseHaplotypeArray(genotypes=geno, variants=variants)

    @staticmethod
    def two_chrom(n=500, m_per_chrom=50, seed=42) -> DenseHaplotypeArray:
        """Haplotypes spread across two chromosomes."""
        m = m_per_chrom * 2
        rng = np.random.RandomState(seed)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        chrom = np.array([1]*m_per_chrom + [2]*m_per_chrom)
        variants = VariantMeta(vid=np.arange(m), chrom=chrom)
        return DenseHaplotypeArray(genotypes=geno, variants=variants)


class TestEffects:
    """Factory for test effect specifications."""

    @staticmethod
    def additive(m=100, h2=0.5, seed=42) -> AdditiveEffects:
        return AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)

    @staticmethod
    def multivariate(m=100, h2=None, rg=0.2, seed=42) -> MultivariateEffects:
        if h2 is None:
            h2 = [0.5, 0.3]
        return MultivariateEffects.from_h2_rg(h2=h2, rg=rg, m=m, seed=seed)

    @staticmethod
    def sparse(m=100, h2=0.5, k_causal=10, seed=42) -> SparseEffects:
        return SparseEffects.from_h2(h2=h2, m=m, k_causal=k_causal, seed=seed)


class TestMeta:
    """Factory for test metadata objects."""

    @staticmethod
    def samples(n=500, n_fam=100, seed=42) -> SampleMeta:
        rng = np.random.RandomState(seed)
        iid = np.arange(n)
        fid = np.repeat(np.arange(n_fam), (n + n_fam - 1) // n_fam)[:n]
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        return SampleMeta(iid=iid, fid=fid, sex=sex)

    @staticmethod
    def variants(m=100, n_chrom=2) -> VariantMeta:
        vid = np.arange(m)
        chrom_size = m // n_chrom
        chrom = np.repeat(np.arange(1, n_chrom + 1), chrom_size)
        # Handle remainder
        if len(chrom) < m:
            chrom = np.concatenate([chrom, np.full(m - len(chrom), n_chrom)])
        return VariantMeta(vid=vid, chrom=chrom[:m])
