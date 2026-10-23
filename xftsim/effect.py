"""
Effect specification classes for the new architecture system.

EffectSpec ABC and concrete implementations:
- AdditiveEffects: univariate additive genetic effects
- MultivariateEffects: multivariate correlated effects across traits
- SparseEffects: sparse causal effects (subset of variants)
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from scipy.optimize import nnls


def compute_maf_ldscore(genotypes: np.ndarray,
                        chrom: Optional[np.ndarray] = None,
                        bias_correct: bool = True) -> tuple:
    """Compute per-SNP minor allele frequency and LD score from genotypes.

    Produces the ``maf`` and ``ldscore`` annotations consumed by
    :meth:`MultivariateEffects.from_frequency_dependent`. LD scores are
    computed *in sample* as the sum of squared genotype correlations of each
    SNP with the SNPs in its block (``ell_j = sum_k r_jk^2``), following
    Bulik-Sullivan et al. (2015). When ``bias_correct`` is True the standard
    finite-sample adjustment ``r2 -> r2 - (1 - r2) / (n - 2)`` is applied to
    remove the upward bias of ``r2`` (so an unlinked SNP has expected LD
    score 1, its own self term).

    Correlations are computed only *within* the blocks defined by ``chrom``
    (SNPs on different chromosomes are treated as unlinked). For a panel of
    independent founder haplotypes (e.g. uniform-AF founders) every SNP's LD
    score is ~1 up to sampling noise.

    Parameters
    ----------
    genotypes : np.ndarray
        (n, m) diploid genotype matrix (allele counts 0/1/2), individuals by
        variants. From a ``HaplotypeArray`` ``h`` use the diploid genotype
        matrix (e.g. summing the two haplotype copies).
    chrom : np.ndarray, optional
        (m,) chromosome label per SNP. If None, all SNPs form one block.
    bias_correct : bool
        Apply the ``r2 - (1 - r2) / (n - 2)`` finite-sample correction.

    Returns
    -------
    maf : np.ndarray
        (m,) minor allele frequencies in [0, 0.5].
    ldscore : np.ndarray
        (m,) LD scores. Monomorphic SNPs get LD score 0.

    Raises
    ------
    ValueError
        If fewer than 3 individuals (needed for the bias correction / r2).
    """
    G = np.asarray(genotypes, dtype=np.float64)
    n, m = G.shape
    if n < 3:
        raise ValueError(f"need >= 3 individuals to compute LD scores, got {n}")

    p = G.mean(axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)

    sd = G.std(axis=0)
    mono = sd == 0.0
    sd_safe = np.where(mono, 1.0, sd)
    Z = (G - G.mean(axis=0)) / sd_safe        # standardized; monomorphic cols ~ 0
    Z[:, mono] = 0.0

    if chrom is None:
        blocks = [np.arange(m)]
    else:
        chrom = np.asarray(chrom)
        blocks = [np.where(chrom == c)[0] for c in np.unique(chrom)]

    ldscore = np.zeros(m, dtype=np.float64)
    for idx in blocks:
        Zb = Z[:, idx]
        r2 = (Zb.T @ Zb / n) ** 2             # (b, b) squared correlations
        if bias_correct:
            r2 = r2 - (1.0 - r2) / (n - 2)
        ldscore[idx] = r2.sum(axis=1)
    ldscore[mono] = 0.0
    return maf, ldscore


def compute_ldak_weights(genotypes: np.ndarray,
                         chrom: Optional[np.ndarray] = None,
                         bias_correct: bool = True) -> tuple:
    """Compute per-SNP minor allele frequency and LDAK weights from genotypes.

    Produces the ``maf`` and ``ldak_weight`` annotations consumed by the LDAK
    heritability model (:meth:`MultivariateEffects.from_ldak_model` /
    :meth:`AdditiveEffects.from_ldak_model`), following Speed et al.
    (2020, *Nat Commun* 11:3865) and the original LDAK weighting of Speed et
    al. (2012). The LDAK weight downweights SNPs that are redundantly tagged
    by their neighbours, so isolated (low-LD) SNPs carry more weight.

    This is a *different* annotation from :func:`compute_maf_ldscore`. The LD
    score ``ell_j = sum_k r_jk^2`` is **high** for SNPs in high LD; the LDAK
    weight ``w_j`` is **high** for SNPs in low LD. Using an LD score where an
    LDAK weight is required reverses the intended tilt of the model.

    Weights are obtained, within each block defined by ``chrom``, as the
    non-negative least-squares solution of ``R2 @ w = 1`` where ``R2`` is the
    matrix of squared genotype correlations in the block (diagonal set to 1,
    i.e. each SNP tags itself). Intuition: ``w`` is chosen so every SNP's
    neighbourhood is tagged ~evenly. Perfectly independent SNPs give
    ``R2 = I`` and hence ``w = 1`` for every SNP (LDAK and GCTA coincide);
    a set of ``d`` perfectly correlated SNPs shares a total weight of 1.

    Parameters
    ----------
    genotypes : np.ndarray
        (n, m) diploid genotype matrix (allele counts 0/1/2), individuals by
        variants (e.g. ``haplotypes.diploid_genotypes``).
    chrom : np.ndarray, optional
        (m,) chromosome label per SNP. If None, all SNPs form one block.
        LD/weighting is computed only *within* a block.
    bias_correct : bool
        Apply the ``r2 -> r2 - (1 - r2) / (n - 2)`` finite-sample correction
        to the off-diagonal squared correlations before solving.

    Returns
    -------
    maf : np.ndarray
        (m,) minor allele frequencies in [0, 0.5].
    ldak_weight : np.ndarray
        (m,) LDAK weights, >= 0. Monomorphic SNPs get weight 0.

    Raises
    ------
    ValueError
        If fewer than 3 individuals (needed for the bias correction / r2).

    Notes
    -----
    The per-block solve is ``O(b^3)`` in the block size ``b`` (dense NNLS), so
    very large single blocks are expensive; pass ``chrom`` (or pre-thin) to
    keep blocks tractable. For a panel of independent founder haplotypes every
    weight is ~1 up to sampling noise.
    """
    G = np.asarray(genotypes, dtype=np.float64)
    n, m = G.shape
    if n < 3:
        raise ValueError(f"need >= 3 individuals to compute LDAK weights, got {n}")

    p = G.mean(axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)

    sd = G.std(axis=0)
    mono = sd == 0.0
    sd_safe = np.where(mono, 1.0, sd)
    Z = (G - G.mean(axis=0)) / sd_safe        # standardized; monomorphic cols ~ 0
    Z[:, mono] = 0.0

    if chrom is None:
        blocks = [np.arange(m)]
    else:
        chrom = np.asarray(chrom)
        blocks = [np.where(chrom == c)[0] for c in np.unique(chrom)]

    weights = np.zeros(m, dtype=np.float64)
    for idx in blocks:
        poly = idx[~mono[idx]]                # skip monomorphic SNPs (weight 0)
        if poly.size == 0:
            continue
        Zb = Z[:, poly]
        r2 = (Zb.T @ Zb / n) ** 2             # (b, b) squared correlations
        if bias_correct:
            r2 = r2 - (1.0 - r2) / (n - 2)
        np.fill_diagonal(r2, 1.0)             # each SNP tags itself
        w, _ = nnls(r2, np.ones(poly.size))   # argmin ||R2 w - 1|| s.t. w >= 0
        weights[poly] = w
    weights[mono] = 0.0
    return maf, weights


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

    @classmethod
    def _from_variance_model(cls, maf: np.ndarray, annotation: np.ndarray,
                             h2: float, a: float, b: float,
                             standardized: bool, seed: Optional[int]) -> "AdditiveEffects":
        """Draw univariate effects with per-SNP variance ``h2 * w / sum(w)``,
        where ``w_j = annotation_j ** b * [2 f_j (1 - f_j)] ** a`` on the
        standardized genotype scale. Shared engine for the GCTA/LDAK models.
        """
        maf = np.asarray(maf, dtype=np.float64)
        annotation = np.asarray(annotation, dtype=np.float64)
        m = len(maf)
        if len(annotation) != m:
            raise ValueError(
                f"maf ({m}) and annotation ({len(annotation)}) must have equal length")
        het = 2.0 * maf * (1.0 - maf)
        W = (annotation ** b) * (het ** a)
        W = np.where(maf > 0.0, W, 0.0)       # monomorphic SNPs carry no heritability
        total = W.sum()
        if total <= 0.0:
            raise ValueError("variance weights sum to zero (no polymorphic SNPs?)")
        sigma2 = h2 * W / total               # per-SNP variance, sums to h2
        rng = np.random.RandomState(seed)
        effects = rng.normal(0.0, 1.0, size=m) * np.sqrt(sigma2)
        inst = cls(effects=effects, standardized=standardized,
                   variant_mask=np.ones(m, dtype=bool))
        inst.design_h2 = h2
        return inst

    @classmethod
    def from_gcta_model(cls, maf: np.ndarray, h2: float,
                        standardized: bool = True,
                        seed: Optional[int] = None) -> "AdditiveEffects":
        """Univariate effects under the GCTA heritability model.

        The GCTA model (Yang et al. 2011; Speed et al. 2020, Eq. 12 with
        ``a = b = 0``) assigns **equal expected heritability to every causal
        SNP**, independent of MAF or LD. On the standardized scale this is
        ``Var(beta_j) = h2 / m`` for all ``j`` -- identical to
        :meth:`from_h2`, but named so the modelling assumption is explicit.

        Parameters
        ----------
        maf : np.ndarray
            (m,) minor allele frequencies (used only to drop monomorphic SNPs;
            the per-SNP variance itself is flat).
        h2 : float
            Target heritability; realized design h2 ``sum_j Var(beta_j) == h2``.
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        AdditiveEffects
            With extra attribute ``design_h2``.
        """
        maf = np.asarray(maf, dtype=np.float64)
        return cls._from_variance_model(maf, np.ones(len(maf)), h2=h2,
                                        a=0.0, b=0.0, standardized=standardized,
                                        seed=seed)

    @classmethod
    def from_ldak_model(cls, maf: np.ndarray, ldak_weight: np.ndarray, h2: float,
                        a: float = 0.75, b: float = 1.0,
                        standardized: bool = True,
                        seed: Optional[int] = None) -> "AdditiveEffects":
        """Univariate effects under the LDAK heritability model.

        The LDAK model (Speed et al. 2020, Eq. 12) sets the per-SNP expected
        heritability (standardized scale) to::

            Var(beta_j)  proportional to  w_j ** b  *  [2 f_j (1 - f_j)] ** a

        with the paper's defaults ``a = 0.75, b = 1``. Here ``w_j`` is the
        **LDAK weight** from :func:`compute_ldak_weights` (not an LD score).
        ``a = b = 0`` recovers the GCTA model.

        Parameters
        ----------
        maf : np.ndarray
            (m,) minor allele frequencies.
        ldak_weight : np.ndarray
            (m,) LDAK weights (see :func:`compute_ldak_weights`).
        h2 : float
            Target heritability; realized design h2 sums to ``h2``.
        a : float
            MAF-coupling exponent on ``[2 f (1 - f)]`` (paper's ``a``,
            default 0.75). Relates to the per-allele ``alpha`` by ``a = alpha + 1``.
        b : float
            LD exponent on the LDAK weight (paper's ``b``, default 1.0).
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        AdditiveEffects
            With extra attribute ``design_h2``.
        """
        return cls._from_variance_model(maf, ldak_weight, h2=h2, a=a, b=b,
                                        standardized=standardized, seed=seed)

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
    def from_frequency_dependent(cls, maf: np.ndarray, ldscore: np.ndarray,
                                 h2: list, R, alpha, s=0.0,
                                 standardized: bool = True,
                                 seed: Optional[int] = None) -> "MultivariateEffects":
        """Generate multivariate effects with MAF/LD-dependent per-SNP variances.

        Generalizes :meth:`from_h2_rg` so that each trait's per-SNP effect
        variance depends on the SNP's minor allele frequency and LD score,
        following the LDAK/RHE-mc frequency- and LD-dependent architecture
        (Speed et al. 2017; Pazokitoroudi et al. 2020, Nat Commun, Eq. 12).

        For trait ``t`` and SNP ``j`` the (unnormalized) variance weight is::

            w_t(j) = [2 f_j (1 - f_j)] ** (1 + alpha_t)  *  ldscore_j ** s_t

        The exponent is ``1 + alpha`` rather than ``alpha`` because effects here
        are on the **standardized** genotype scale (see ``standardized``): the
        standardized effect is ``beta = a * sqrt(2 f (1 - f))`` for a per-allele
        effect ``a`` with ``Var(a) ~ [2 f (1 - f)] ** alpha``, which contributes
        the extra power of the genotype variance. Consequently ``alpha = -1,
        s = 0`` recovers the flat GCTA model and reproduces ``from_h2_rg``.

        Weights are normalized per trait so that ``sum_j Var(beta_tj) = h2_t``,
        i.e. the design heritability matches the target (cf. ``from_h2``).

        Cross-trait structure: ``R`` (or scalar ``rg``) is the **per-SNP** effect
        correlation matrix. When traits share the same ``alpha`` and ``s`` this
        equals the aggregate genetic correlation; when they differ, the realized
        aggregate is attenuated (Cauchy-Schwarz) and is exposed on the returned
        object as ``.realized_rg`` for inspection.

        Parameters
        ----------
        maf : np.ndarray
            (m,) minor allele frequencies, one per SNP.
        ldscore : np.ndarray
            (m,) LD scores, one per SNP. Ignored when all ``s`` are 0.
        h2 : list
            (k,) per-trait target heritabilities.
        R : np.ndarray or float
            (k, k) per-SNP genetic correlation matrix, or a scalar ``rg``
            applied to every off-diagonal pair.
        alpha : float or array-like
            Per-trait MAF-scaling exponent. Scalar broadcasts to all traits.
            ``alpha = -1`` removes MAF dependence (equal per-SNP heritability).
        s : float or array-like, optional
            Per-trait LD-score exponent. Scalar broadcasts to all traits.
            Default 0 (no LD dependence).
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        MultivariateEffects
            With extra attributes ``design_h2`` (k,) and ``realized_rg`` (k, k).

        Raises
        ------
        ValueError
            If array shapes are inconsistent or ``R`` is not (k, k).
        """
        h2 = np.asarray(h2, dtype=np.float64)
        k = len(h2)
        maf = np.asarray(maf, dtype=np.float64)
        ldscore = np.asarray(ldscore, dtype=np.float64)
        m = len(maf)
        if len(ldscore) != m:
            raise ValueError(f"maf ({m}) and ldscore ({len(ldscore)}) must have equal length")

        alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (k,))
        s = np.broadcast_to(np.asarray(s, dtype=np.float64), (k,))

        # Correlation matrix R (k, k): build from scalar rg if needed.
        if np.isscalar(R):
            R = np.full((k, k), float(R))
            np.fill_diagonal(R, 1.0)
        else:
            R = np.asarray(R, dtype=np.float64)
            if R.shape != (k, k):
                raise ValueError(f"R must be ({k}, {k}), got {R.shape}")

        # Per-SNP, per-trait variance weights w_t(j), shape (m, k).
        het = (2.0 * maf * (1.0 - maf))[:, None]          # (m, 1)
        W = het ** (1.0 + alpha)[None, :] * ldscore[:, None] ** s[None, :]
        # Normalize each trait's weights to sum to its target h2.
        colsum = W.sum(axis=0)                            # (k,)
        sigma2 = h2[None, :] * W / colsum[None, :]        # (m, k), columns sum to h2_t
        sd = np.sqrt(sigma2)                              # (m, k)

        # Draw z_j ~ MVN(0, R) (shared R), then scale rows: beta_j = sd_j * z_j
        # gives beta_j ~ MVN(0, diag(sd_j) R diag(sd_j)) per SNP.
        rng = np.random.RandomState(seed)
        Z = rng.multivariate_normal(np.zeros(k), R, size=m)   # (m, k)
        effects = sd * Z                                      # (m, k)

        inst = cls(effects=effects, standardized=standardized,
                   variant_mask=np.ones(m, dtype=bool))

        # Design-level realized aggregate genetic correlation:
        #   G[a,b] = R[a,b] * sum_j sd_ja sd_jb ; realized_rg = G / sqrt(diag outer)
        G = R * (sd.T @ sd)                               # (k, k); diag = h2
        d = np.sqrt(np.diag(G))
        inst.realized_rg = G / np.outer(d, d)
        inst.design_h2 = h2
        return inst

    @classmethod
    def from_gcta_model(cls, maf: np.ndarray, h2: list, R=0.0,
                        standardized: bool = True,
                        seed: Optional[int] = None) -> "MultivariateEffects":
        """Multivariate effects under the GCTA heritability model.

        Every causal SNP contributes equal expected heritability to each trait
        (Speed et al. 2020, Eq. 12 with ``a = b = 0``); on the standardized
        scale ``Var(beta_tj) = h2_t / m``. Thin wrapper over
        :meth:`from_frequency_dependent` with ``alpha = -1, s = 0`` (so the LD
        annotation is irrelevant), exposed so the model is explicit.

        Parameters
        ----------
        maf : np.ndarray
            (m,) minor allele frequencies (used only to size the panel).
        h2 : list
            (k,) per-trait target heritabilities.
        R : np.ndarray or float
            (k, k) per-SNP genetic correlation matrix, or scalar ``rg``.
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        MultivariateEffects
            With extra attributes ``design_h2`` and ``realized_rg``.
        """
        maf = np.asarray(maf, dtype=np.float64)
        ones = np.ones(len(maf), dtype=np.float64)
        return cls.from_frequency_dependent(maf, ones, h2=h2, R=R,
                                            alpha=-1.0, s=0.0,
                                            standardized=standardized, seed=seed)

    @classmethod
    def from_ldak_model(cls, maf: np.ndarray, ldak_weight: np.ndarray, h2: list,
                        R=0.0, a=0.75, b=1.0,
                        standardized: bool = True,
                        seed: Optional[int] = None) -> "MultivariateEffects":
        """Multivariate effects under the LDAK heritability model.

        Per-SNP expected heritability (standardized scale) for trait ``t``::

            Var(beta_tj)  proportional to  w_j ** b_t  *  [2 f_j (1 - f_j)] ** a_t

        with the paper's defaults ``a = 0.75, b = 1`` (Speed et al. 2020,
        Eq. 12). ``w_j`` is the **LDAK weight** (:func:`compute_ldak_weights`),
        *not* an LD score. ``a = b = 0`` recovers :meth:`from_gcta_model`.

        Implemented via :meth:`from_frequency_dependent` using the mapping
        ``alpha = a - 1`` (per-allele exponent) and ``s = b`` with the LDAK
        weight supplied as the LD annotation. ``a`` and ``b`` may be scalars
        (shared across traits) or length-``k`` arrays.

        Parameters
        ----------
        maf : np.ndarray
            (m,) minor allele frequencies.
        ldak_weight : np.ndarray
            (m,) LDAK weights (see :func:`compute_ldak_weights`).
        h2 : list
            (k,) per-trait target heritabilities.
        R : np.ndarray or float
            (k, k) per-SNP genetic correlation matrix, or scalar ``rg``.
        a : float or array-like
            MAF-coupling exponent (paper's ``a``, default 0.75). Per-trait if
            array-like. Relates to per-allele ``alpha`` by ``a = alpha + 1``.
        b : float or array-like
            LD exponent on the LDAK weight (paper's ``b``, default 1.0).
        standardized : bool
            Whether effects are for standardized genotypes.
        seed : int, optional
            Random seed.

        Returns
        -------
        MultivariateEffects
            With extra attributes ``design_h2`` and ``realized_rg``.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return cls.from_frequency_dependent(maf, ldak_weight, h2=h2, R=R,
                                            alpha=a - 1.0, s=b,
                                            standardized=standardized, seed=seed)

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
