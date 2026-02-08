"""
Effect specification classes for the new architecture system.

EffectSpec ABC and concrete implementations:
- AdditiveEffects: univariate additive genetic effects
- MultivariateEffects: multivariate correlated effects across traits
- SparseEffects: sparse causal effects (subset of variants)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class EffectSpec(ABC):
    """
    Abstract base class for genetic effect specifications.

    Effects are sampled once at architecture creation and fixed for all generations.
    The HaplotypeOperator picks the right matvec based on the `standardized` flag.

    Attributes
    ----------
    effects : np.ndarray
        Effect sizes — shape (m,) for univariate, (m, k) for multivariate.
    standardized : bool
        If True, effects are for standardized genotypes (use standardized_matvec).
    variant_mask : np.ndarray
        Boolean array indicating which variants are causal.
    """

    def __init__(self, effects: np.ndarray, standardized: bool, variant_mask: np.ndarray):
        self.effects = np.asarray(effects, dtype=np.float64)
        self.standardized = standardized
        self.variant_mask = np.asarray(variant_mask, dtype=bool)

    @property
    def m(self) -> int:
        """Total number of variants (causal + non-causal)."""
        return len(self.variant_mask)

    @property
    def m_causal(self) -> int:
        """Number of causal variants."""
        return int(np.sum(self.variant_mask))

    @property
    def k(self) -> int:
        """Number of traits (1 for univariate)."""
        if self.effects.ndim == 1:
            return 1
        return self.effects.shape[1]


class AdditiveEffects(EffectSpec):
    """Univariate additive genetic effects.

    All variants are causal (variant_mask is all-True). Effect sizes
    are stored as a 1D array of shape (m,).

    Examples
    --------
    Generate effects targeting h2 = 0.5 across 100 variants:

    >>> eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
    >>> eff.m
    100
    >>> eff.k
    1

    Create from a known effect array:

    >>> import numpy as np
    >>> eff = AdditiveEffects.from_array(np.array([0.1, -0.2, 0.05]))
    >>> eff.m
    3
    """

    @classmethod
    def from_h2(cls, h2: float, m: int, standardized: bool = True,
                seed: Optional[int] = None) -> "AdditiveEffects":
        """Generate additive effects to target a given heritability.

        Under standardized genotypes, Var(G) = sum(beta^2).
        We draw beta ~ N(0, h2/m) so E[sum(beta^2)] = h2.

        Parameters
        ----------
        h2 : float
            Target heritability.
        m : int
            Number of variants (all causal).
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        AdditiveEffects
        """
        rng = np.random.RandomState(seed)
        effects = rng.normal(0, np.sqrt(h2 / m), size=m)
        variant_mask = np.ones(m, dtype=bool)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    @classmethod
    def from_array(cls, effects: np.ndarray, standardized: bool = True) -> "AdditiveEffects":
        """Create from a pre-specified effect array.

        Parameters
        ----------
        effects : np.ndarray
            (m,) array of effect sizes.
        standardized : bool
            Whether effects are for standardized genotypes.

        Returns
        -------
        AdditiveEffects
        """
        effects = np.asarray(effects, dtype=np.float64)
        variant_mask = np.ones(len(effects), dtype=bool)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    def __repr__(self) -> str:
        return (f"AdditiveEffects(m={self.m}, m_causal={self.m_causal}, "
                f"standardized={self.standardized})")


class MultivariateEffects(EffectSpec):
    """Multivariate correlated effects across k traits.

    Effect sizes are stored as a 2D array of shape (m, k) where k is the
    number of traits. All variants are causal.
    """

    @classmethod
    def from_h2_rg(cls, h2: list, rg: float, m: int,
                   standardized: bool = True,
                   seed: Optional[int] = None) -> "MultivariateEffects":
        """
        Generate multivariate effects from heritabilities and genetic correlation.

        Parameters
        ----------
        h2 : list
            Per-trait heritabilities [h2_1, h2_2, ...].
        rg : float
            Genetic correlation between traits.
        m : int
            Number of variants (all causal).
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        MultivariateEffects
        """
        h2 = np.asarray(h2, dtype=np.float64)
        k = len(h2)
        # Build per-SNP covariance: Cov_snp = diag(sqrt(h2/m)) @ R @ diag(sqrt(h2/m))
        sd = np.sqrt(h2 / m)
        R = np.full((k, k), rg)
        np.fill_diagonal(R, 1.0)
        cov_snp = np.outer(sd, sd) * R
        rng = np.random.RandomState(seed)
        effects = rng.multivariate_normal(np.zeros(k), cov_snp, size=m)
        variant_mask = np.ones(m, dtype=bool)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    @classmethod
    def from_covg(cls, covg: np.ndarray, m: int,
                  standardized: bool = True,
                  seed: Optional[int] = None) -> "MultivariateEffects":
        """Generate multivariate effects from a full genetic covariance matrix.

        Parameters
        ----------
        covg : np.ndarray
            (k, k) genetic covariance matrix. sum(beta @ beta.T) approx covg.
        m : int
            Number of variants.
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        MultivariateEffects
        """
        covg = np.asarray(covg, dtype=np.float64)
        cov_snp = covg / m
        rng = np.random.RandomState(seed)
        k = covg.shape[0]
        effects = rng.multivariate_normal(np.zeros(k), cov_snp, size=m)
        variant_mask = np.ones(m, dtype=bool)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    @classmethod
    def from_array(cls, effects: np.ndarray,
                   standardized: bool = True) -> "MultivariateEffects":
        """Create from a pre-specified (m, k) effect matrix.

        Parameters
        ----------
        effects : np.ndarray
            (m, k) array of effect sizes.
        standardized : bool
            Whether effects are for standardized genotypes.

        Returns
        -------
        MultivariateEffects
        """
        effects = np.asarray(effects, dtype=np.float64)
        variant_mask = np.ones(effects.shape[0], dtype=bool)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    def __repr__(self) -> str:
        return (f"MultivariateEffects(m={self.m}, k={self.k}, "
                f"m_causal={self.m_causal}, standardized={self.standardized})")


class SparseEffects(EffectSpec):
    """Sparse causal effects -- only a subset of variants are causal.

    Non-causal variants have zero effect sizes. The ``variant_mask``
    indicates which variants are causal.
    """

    @classmethod
    def from_h2(cls, h2: float, m: int, k_causal: int,
                standardized: bool = True,
                seed: Optional[int] = None) -> "SparseEffects":
        """
        Generate sparse additive effects.

        Parameters
        ----------
        h2 : float
            Target heritability.
        m : int
            Total number of variants.
        k_causal : int
            Number of causal variants.
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        SparseEffects

        Raises
        ------
        ValueError
            If ``k_causal > m``.
        """
        if k_causal > m:
            raise ValueError(f"k_causal ({k_causal}) > m ({m})")
        rng = np.random.RandomState(seed)
        # Select causal variants
        causal_idx = rng.choice(m, size=k_causal, replace=False)
        causal_idx.sort()
        variant_mask = np.zeros(m, dtype=bool)
        variant_mask[causal_idx] = True
        # Effects: (m,) with zeros for non-causal
        effects = np.zeros(m, dtype=np.float64)
        effects[causal_idx] = rng.normal(0, np.sqrt(h2 / k_causal), size=k_causal)
        return cls(effects=effects, standardized=standardized, variant_mask=variant_mask)

    def __repr__(self) -> str:
        return (f"SparseEffects(m={self.m}, m_causal={self.m_causal}, "
                f"standardized={self.standardized})")
