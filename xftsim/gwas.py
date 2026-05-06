"""
GWAS and Polygenic Score (PGS) computation from simulation output.

Provides:
- GWAS: per-variant association statistics (beta, SE, t, p) for each phenotype
- PGS: polygenic score calculation from external weights
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from xftsim.struct import HaplotypeOperator, DenseHaplotypeArray, NPhenotypeArray


@dataclass
class GWASResult:
    """
    Per-variant association statistics for a single phenotype.

    Attributes
    ----------
    beta : np.ndarray
        (m,) regression coefficients (phenotype on genotype dosage).
    se : np.ndarray
        (m,) standard errors of beta.
    t_stat : np.ndarray
        (m,) t-statistics (beta / se).
    p_value : np.ndarray
        (m,) two-sided p-values.
    af : np.ndarray
        (m,) allele frequencies.
    n : int
        Sample size used.
    """
    beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    af: np.ndarray
    n: int


class GWAS:
    """
    Genome-wide association study: per-variant regression of phenotype on
    genotype dosage.

    Computes OLS regression statistics for every variant simultaneously
    using efficient matrix operations (no per-variant loops).

    Parameters
    ----------
    haplotypes : HaplotypeOperator
        Genotype data (DenseHaplotypeArray or GraphHaplotypeOperator).
    phenotypes : NPhenotypeArray
        Phenotype data (dict-like, keyed by phenotype name).
    sample_indices : np.ndarray, optional
        Indices of samples to include (e.g. from an UnrelatedView).
        If None, all samples are used.

    Examples
    --------
    >>> gwas = GWAS(haplotypes, phenotypes)
    >>> results = gwas.run()       # all phenotype keys
    >>> results = gwas.run(['Y'])   # specific keys
    >>> r = results['Y']
    >>> r.beta, r.se, r.p_value
    """

    def __init__(self, haplotypes: HaplotypeOperator,
                 phenotypes: NPhenotypeArray,
                 sample_indices: np.ndarray | None = None) -> None:
        self.haplotypes: HaplotypeOperator = haplotypes
        self.phenotypes: NPhenotypeArray = phenotypes
        self.sample_indices: np.ndarray | None = sample_indices

    def run(self, keys: list[str] | None = None) -> dict[str, GWASResult]:
        """
        Run GWAS for specified phenotype keys.

        Parameters
        ----------
        keys : list of str, optional
            Phenotype keys to analyze. If None, uses all keys.

        Returns
        -------
        dict[str, GWASResult]
            Mapping from phenotype key to association results.
        """
        if keys is None:
            keys = list(self.phenotypes.keys)

        # Materialize genotype dosage matrix
        hap = self.haplotypes
        if isinstance(hap, DenseHaplotypeArray):
            G = hap.diploid_genotypes.astype(np.float64)
        else:
            G = hap.to_dense().diploid_genotypes.astype(np.float64)

        # Subset samples if requested
        if self.sample_indices is not None:
            idx = self.sample_indices
            G = G[idx, :]
        else:
            idx = None

        n, m = G.shape

        # Allele frequencies
        af = G.mean(axis=0) / 2.0

        # Precompute per-variant sums for OLS
        # For simple linear regression y ~ x:
        #   beta_j = Cov(x_j, y) / Var(x_j)
        #   SE = sqrt(MSE / SS_x)
        # Vectorized across all m variants.
        G_mean = G.mean(axis=0)  # (m,)
        G_centered = G - G_mean  # (n, m)
        ss_x = np.sum(G_centered ** 2, axis=0)  # (m,)

        results = {}
        for key in keys:
            y = self.phenotypes[key]
            if idx is not None:
                y = y[idx]
            y = np.asarray(y, dtype=np.float64)

            y_mean = y.mean()
            y_centered = y - y_mean  # (n,)

            # Cross-products: sum( (G_j - mean_j) * (y - mean_y) ) for each j
            cross = G_centered.T @ y_centered  # (m,)

            # Regression coefficient
            beta = np.where(ss_x > 0, cross / ss_x, 0.0)

            # Residuals for each variant: r_ij = y_i - (alpha_j + beta_j * G_ij)
            # But computing n x m residuals is expensive. Instead use the identity:
            # MSE_j = (SS_y - beta_j^2 * SS_x_j) / (n - 2)
            ss_y = np.sum(y_centered ** 2)

            # Residual sum of squares per variant
            ss_res = ss_y - beta ** 2 * ss_x  # (m,)
            ss_res = np.maximum(ss_res, 0.0)  # numerical guard

            df = n - 2
            if df <= 0:
                # Degenerate case: not enough samples
                se = np.full(m, np.nan)
                t_stat = np.full(m, np.nan)
                p_value = np.full(m, np.nan)
            else:
                mse = ss_res / df  # (m,)
                se = np.where(ss_x > 0, np.sqrt(mse / ss_x), np.nan)
                t_stat = np.where(se > 0, beta / se, 0.0)
                # Two-sided p-value from t-distribution
                p_value = np.where(
                    np.isfinite(t_stat),
                    2.0 * stats.t.sf(np.abs(t_stat), df=df),
                    np.nan,
                )

            results[key] = GWASResult(
                beta=beta,
                se=se,
                t_stat=t_stat,
                p_value=p_value,
                af=af,
                n=n,
            )

        return results


class PGS:
    """
    Polygenic score calculator.

    Computes PGS = haplotypes @ weights using the HaplotypeOperator's
    matvec (raw or standardized).

    Parameters
    ----------
    haplotypes : HaplotypeOperator
        Genotype data.
    weights : np.ndarray or dict
        Effect weights. If dict, maps variant IDs to weights (looked up
        against haplotypes.variants.vid). If np.ndarray, must be (m,).
    standardized : bool
        If True, use standardized genotypes (centered by 2*AF).
        Default is False (raw dosage).

    Examples
    --------
    >>> pgs = PGS(haplotypes, weights=beta_hat)
    >>> scores = pgs.score()
    """

    def __init__(self, haplotypes: HaplotypeOperator,
                 weights: np.ndarray | dict[str, float],
                 standardized: bool = False) -> None:
        self.haplotypes: HaplotypeOperator = haplotypes
        self.standardized: bool = standardized

        if isinstance(weights, dict):
            # Look up variant IDs and build weight vector
            vid = self.haplotypes.variants.vid
            w = np.zeros(len(vid), dtype=np.float64)
            vid_list = list(vid)
            for v, wt in weights.items():
                try:
                    j = vid_list.index(v)
                    w[j] = wt
                except ValueError:
                    pass  # variant not in haplotypes, skip
            self.weights = w
        else:
            self.weights = np.asarray(weights, dtype=np.float64)

    def score(self) -> np.ndarray:
        """
        Compute polygenic scores.

        Returns
        -------
        np.ndarray
            (n,) array of polygenic scores, one per sample.
        """
        if self.standardized:
            return self.haplotypes.standardized_matvec(self.weights)
        else:
            return self.haplotypes.matvec(self.weights)
