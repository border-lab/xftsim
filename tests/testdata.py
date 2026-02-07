"""
Deterministic test data generators for xftsim tests.

All generators use explicit seeds for reproducibility.
Usable both inside pytest (via conftest fixtures) and standalone.
"""
import numpy as np
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent,
    AggregationComponent, ParentComponent,
)
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap


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


class TestSimulation:
    """Factory for simulation test fixtures."""

    @staticmethod
    def founder_haplotypes(n=500, m=50, seed=42) -> DenseHaplotypeArray:
        """Founder haplotypes with balanced sex and known AFs."""
        rng = np.random.RandomState(seed)
        af = rng.uniform(0.1, 0.9, size=m)
        geno = np.zeros((n, m, 2), dtype=np.int8)
        for j in range(m):
            geno[:, j, 0] = rng.binomial(1, af[j], size=n)
            geno[:, j, 1] = rng.binomial(1, af[j], size=n)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=af)
        return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)

    @staticmethod
    def simple_architecture(m=50, h2=0.5, seed=123) -> Architecture:
        """Single-trait architecture: Y = G + E, Var(G)~h2, Var(E)~(1-h2)."""
        effects = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        return arch

    @staticmethod
    def recombination_map(m=50, p=0.5) -> RecombinationMap:
        """Constant recombination map."""
        return RecombinationMap.constant_map(m=m, p=p)

    @staticmethod
    def mating_regime(offspring_per_pair=2) -> RandomMating:
        """Default random mating."""
        return RandomMating(offspring_per_pair=offspring_per_pair)

    @staticmethod
    def bivariate_architecture(m=50, h2=None, rg=0.2, seed=123) -> Architecture:
        """
        Bivariate architecture: two correlated traits with shared genetic + noise.

        (trait1.G, trait2.G) ~ mvGenetic(eff)
        trait1.E ~ noise(1 - h2[0])
        trait2.E ~ noise(1 - h2[1])
        trait1 ~ trait1.G + trait1.E
        trait2 ~ trait2.G + trait2.E
        """
        if h2 is None:
            h2 = [0.5, 0.3]
        effects = MultivariateEffects.from_h2_rg(h2=h2, rg=rg, m=m, seed=seed)
        arch = Architecture()
        arch.add(['trait1.G', 'trait2.G'], MVGeneticComponent(effects))
        arch.add('trait1.E', NoiseComponent(variance=1.0 - h2[0]))
        arch.add('trait2.E', NoiseComponent(variance=1.0 - h2[1]))
        arch.add('trait1', AggregationComponent('trait1.G + trait1.E'))
        arch.add('trait2', AggregationComponent('trait2.G + trait2.E'))
        return arch

    @staticmethod
    def vt_architecture(m=50, h2=0.5, vt_weight=0.3, seed=123) -> Architecture:
        """
        Architecture with vertical transmission (VT).

        Y.G ~ genetic(eff)
        Y.VT ~ parent(Y)  with founder fallback noise
        Y.E ~ noise(residual_var)
        Y ~ Y.G + vt_weight * Y.VT + Y.E
        """
        effects = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
        residual_var = 1.0 - h2
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.5)))
        arch.add('Y.E', NoiseComponent(variance=residual_var))
        arch.add('Y', AggregationComponent(f'Y.G + {vt_weight} * Y.VT + Y.E'))
        return arch

    @staticmethod
    def assortative_mating_regime(component_names=None, r=0.5,
                                  offspring_per_pair=2) -> LinearAssortativeMating:
        """Assortative mating regime for test fixtures."""
        if component_names is None:
            component_names = ['Y']
        return LinearAssortativeMating(
            component_names=component_names, r=r,
            offspring_per_pair=offspring_per_pair,
        )
