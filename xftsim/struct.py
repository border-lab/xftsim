import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
import dask.array as da
from dataclasses import dataclass, field
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from abc import ABC, abstractmethod
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, Tuple, Optional
from functools import cached_property
from scipy import interpolate
import xftsim as xft


class GeneticMap:
    """
    Map between physical and genetic distances.

    Parameters
    ----------
    chrom : Iterable
        Chromsomes variants are located on
    pos_bp : Iterable
        Physical positions of variants
    pos_cM : Iterable
        Map distances in cM

    Attributes
    __________
    frame : pd.DataFrame
        Pandas DataFrame with the above columns
    chroms : np.ndarray
        Unique chromosomes present in map
    """
    def __init__(self,
                 chrom: Iterable,
                 pos_bp: Iterable,
                 pos_cM: Iterable):
        self.frame = pd.DataFrame.from_dict(dict(chrom = chrom,
                                                 pos_bp = pos_bp,
                                                 pos_cM = pos_cM))
        self.chroms = np.unique(self.frame.chrom.values.astype(int)).astype(str)

    @classmethod
    def from_pyrho_maps(cls, paths: Iterable, sep='\t', **kwargs) -> "GeneticMap":
        """Construct genetic map objects from maps provided at https://github.com/popgenmethods/pyrho
        Please cite their work if you use their maps.
        
        Parameters
        ----------
        paths : Iterable
            Paths for each chromosome
        sep : str, optional
            Passed to pd.read_csv()
        **kwargs
            Additional arguments to pd.read_csv()
        
        Returns
        -------
        GeneticMap
        """
        gmap = pd.concat([pd.read_csv(path, sep='\t', **kwargs) for path in paths])
        chrom = np.char.lstrip(gmap.Chromosome.values.astype(str),'chr')
        pos_bp = gmap['Position(bp)'].values
        pos_cM = gmap['Map(cM)'].values
        return cls(chrom, pos_bp, pos_cM)


    def interpolate_cM_chrom(self, pos_bp: Iterable, chrom: str, **kwargs):
        """
        Interpolate cM values in a specified chromosome based on genetic map information.

        Parameters
        ----------
        pos_bp : Iterable
            Physical positions for which to interpolate cM values
        chrom : str
            Chromosome on which to interpolate
        **kwargs
            Additional keyword arguments to be passed to scipy.interpolate.interp1d.
        """ 
        subset = self.frame[self.frame.chrom==chrom]
        interpolator = interpolate.interp1d(x = subset.pos_bp.values,
                                            y = subset.pos_cM.values,
                                            **kwargs)
        return interpolator(pos_bp)


@dataclass(frozen=True)
class SampleMeta:
    """
    Immutable metadata for samples/individuals.

    Parameters
    ----------
    iid : np.ndarray
        Individual IDs (required).
    fid : np.ndarray, optional
        Family IDs. Defaults to iid if not provided.
    sex : np.ndarray, optional
        Biological sex (0=female, 1=male). Defaults to alternating 0,1.
    generation : int, optional
        Generation number. Default is 0.
    extra : dict, optional
        Arbitrary metadata arrays (ancestry PCs, batch IDs, etc.).

    Attributes
    ----------
    n : int
        Number of individuals.
    n_fam : int
        Number of unique families.
    n_female : int
        Number of biological females (sex=0).
    n_male : int
        Number of biological males (sex=1).
    """
    iid: np.ndarray
    fid: np.ndarray = None
    sex: np.ndarray = None
    generation: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        iid = np.asarray(self.iid)
        object.__setattr__(self, 'iid', iid)

        if self.fid is None:
            object.__setattr__(self, 'fid', iid.copy())
        else:
            object.__setattr__(self, 'fid', np.asarray(self.fid))

        if self.sex is None:
            n = len(iid)
            sex = np.tile([0, 1], (n + 1) // 2)[:n]
            object.__setattr__(self, 'sex', sex)
        else:
            object.__setattr__(self, 'sex', np.asarray(self.sex))

        # Validate and convert extra arrays
        n = len(iid)
        validated = {}
        for k, v in self.extra.items():
            arr = np.asarray(v)
            if len(arr) != n:
                raise ValueError(
                    f"extra['{k}'] has length {len(arr)}, expected {n}"
                )
            validated[k] = arr
        object.__setattr__(self, 'extra', validated)

    @property
    def n(self) -> int:
        """Number of individuals."""
        return len(self.iid)

    @property
    def n_fam(self) -> int:
        """Number of unique families."""
        return len(np.unique(self.fid))

    @property
    def n_female(self) -> int:
        """Number of biological females (sex=0)."""
        return int(np.sum(self.sex == 0))

    @property
    def n_male(self) -> int:
        """Number of biological males (sex=1)."""
        return int(np.sum(self.sex == 1))

    @property
    def unique_identifier(self) -> np.ndarray:
        """
        Unique identifier for each sample, combining generation, iid, and fid.
        Format: '{generation}.{iid}.{fid}'
        """
        return np.array([
            f"{self.generation}.{i}.{f}"
            for i, f in zip(self.iid, self.fid)
        ])

    def subset(self, idx) -> "SampleMeta":
        """Return a new SampleMeta with a subset of samples."""
        return SampleMeta(
            iid=self.iid[idx].copy(),
            fid=self.fid[idx].copy(),
            sex=self.sex[idx].copy(),
            generation=self.generation,
            extra={k: v[idx].copy() for k, v in self.extra.items()},
        )

    def with_generation(self, generation: int) -> "SampleMeta":
        """Return a new SampleMeta with a different generation."""
        return SampleMeta(
            iid=self.iid,
            fid=self.fid,
            sex=self.sex,
            generation=generation,
            extra=self.extra,
        )

    def to_sample_index(self) -> "xft.index.SampleIndex":
        """
        Convert to legacy SampleIndex for compatibility with PhenotypeArray.

        Returns
        -------
        xft.index.SampleIndex
            A SampleIndex with the same data.
        """
        return xft.index.SampleIndex(
            iid=self.iid.astype(str),
            fid=self.fid.astype(str),
            sex=self.sex,
            generation=self.generation,
        )

    def __repr__(self) -> str:
        return (f"SampleMeta(n={self.n}, n_fam={self.n_fam}, "
                f"n_female={self.n_female}, n_male={self.n_male}, "
                f"generation={self.generation})")


@dataclass(frozen=True)
class VariantMeta:
    """
    Immutable metadata for genetic variants.

    Parameters
    ----------
    vid : np.ndarray
        Variant IDs (required).
    chrom : np.ndarray, optional
        Chromosome for each variant.
    pos_bp : np.ndarray, optional
        Base pair position.
    pos_cM : np.ndarray, optional
        Centimorgan position.
    af : np.ndarray, optional
        Allele frequencies.
    zero_allele : np.ndarray, optional
        Reference allele (e.g., 'A').
    one_allele : np.ndarray, optional
        Alternate allele (e.g., 'G').
    extra : dict, optional
        Arbitrary metadata arrays (annotation flags, etc.).

    Attributes
    ----------
    m : int
        Number of variants.
    """
    vid: np.ndarray
    chrom: np.ndarray = None
    pos_bp: np.ndarray = None
    pos_cM: np.ndarray = None
    af: np.ndarray = None
    zero_allele: np.ndarray = None
    one_allele: np.ndarray = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        vid = np.asarray(self.vid)
        object.__setattr__(self, 'vid', vid)
        m = len(vid)

        if self.chrom is not None:
            object.__setattr__(self, 'chrom', np.asarray(self.chrom))
        if self.pos_bp is not None:
            object.__setattr__(self, 'pos_bp', np.asarray(self.pos_bp))
        if self.pos_cM is not None:
            object.__setattr__(self, 'pos_cM', np.asarray(self.pos_cM))
        if self.af is not None:
            object.__setattr__(self, 'af', np.asarray(self.af))
        if self.zero_allele is not None:
            object.__setattr__(self, 'zero_allele', np.asarray(self.zero_allele))
        if self.one_allele is not None:
            object.__setattr__(self, 'one_allele', np.asarray(self.one_allele))

        # Validate and convert extra arrays
        validated = {}
        for k, v in self.extra.items():
            arr = np.asarray(v)
            if len(arr) != m:
                raise ValueError(
                    f"extra['{k}'] has length {len(arr)}, expected {m}"
                )
            validated[k] = arr
        object.__setattr__(self, 'extra', validated)

    def __getitem__(self, key: str) -> np.ndarray:
        """Access core fields or extras by name: variants['coding']."""
        core = {'vid', 'chrom', 'pos_bp', 'pos_cM', 'af', 'zero_allele', 'one_allele'}
        if key in core:
            val = getattr(self, key)
            if val is None:
                raise KeyError(f"Field '{key}' is None")
            return val
        return self.extra[key]

    @property
    def m(self) -> int:
        """Number of variants."""
        return len(self.vid)

    def subset(self, idx) -> "VariantMeta":
        """Return a new VariantMeta with a subset of variants."""
        return VariantMeta(
            vid=self.vid[idx].copy(),
            chrom=self.chrom[idx].copy() if self.chrom is not None else None,
            pos_bp=self.pos_bp[idx].copy() if self.pos_bp is not None else None,
            pos_cM=self.pos_cM[idx].copy() if self.pos_cM is not None else None,
            af=self.af[idx].copy() if self.af is not None else None,
            zero_allele=self.zero_allele[idx].copy() if self.zero_allele is not None else None,
            one_allele=self.one_allele[idx].copy() if self.one_allele is not None else None,
            extra={k: v[idx].copy() for k, v in self.extra.items()},
        )

    def to_variant_index(self, af: np.ndarray = None) -> "xft.index.DiploidVariantIndex":
        """
        Convert to legacy DiploidVariantIndex for compatibility.

        Parameters
        ----------
        af : np.ndarray, optional
            Allele frequencies. If not provided, uses stored af or NaN.

        Returns
        -------
        xft.index.DiploidVariantIndex
            A DiploidVariantIndex with the same data.
        """
        return xft.index.DiploidVariantIndex(
            vid=self.vid,
            chrom=self.chrom,
            pos_bp=self.pos_bp,
            pos_cM=self.pos_cM,
            af=af if af is not None else self.af,
            zero_allele=self.zero_allele,
            one_allele=self.one_allele,
        )

    def __repr__(self) -> str:
        fields = [f"m={self.m}"]
        if self.chrom is not None:
            n_chrom = len(np.unique(self.chrom))
            fields.append(f"n_chrom={n_chrom}")
        if self.af is not None:
            fields.append("af=True")
        return f"VariantMeta({', '.join(fields)})"


# ---------------------------------------------------------------------------
# HaplotypeOperator ABC (new — typed against by Architecture, EffectSpec, etc.)
# ---------------------------------------------------------------------------

class HaplotypeOperator(ABC):
    """Abstract base class for all genotype representations.

    Defines the interface for matrix-vector operations on haplotype data,
    used by the architecture's genetic components. All implementations
    must provide ``samples`` (SampleMeta) and ``variants`` (VariantMeta)
    attributes.

    Concrete implementations:

    - ``DenseHaplotypeArray`` -- NumPy-backed (n, m, 2) array
    - ``GraphHaplotypeOperator`` -- GRG wrapper (pygrgl graph traversal)

    Attributes
    ----------
    samples : SampleMeta
        Sample metadata (set by concrete implementations).
    variants : VariantMeta
        Variant metadata (set by concrete implementations).
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of samples (individuals)."""
        ...

    @property
    @abstractmethod
    def m(self) -> int:
        """Number of diploid variants."""
        ...

    @abstractmethod
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Diploid genotype-vector product: G @ v.

        Parameters
        ----------
        v : np.ndarray
            Effect vector of shape (m,) or (m, k).

        Returns
        -------
        np.ndarray
            Result of shape (n,) or (n, k).
        """
        ...

    @abstractmethod
    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """Transpose genotype-vector product: G.T @ v.

        Parameters
        ----------
        v : np.ndarray
            Vector of shape (n,) or (n, k).

        Returns
        -------
        np.ndarray
            Result of shape (m,) or (m, k).
        """
        ...

    @abstractmethod
    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """Maternal haplotype matvec: hap[:,:,0] @ v.

        Parameters
        ----------
        v : np.ndarray
            Effect vector of shape (m,).

        Returns
        -------
        np.ndarray
            Result of shape (n,).
        """
        ...

    @abstractmethod
    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """Paternal haplotype matvec: hap[:,:,1] @ v.

        Parameters
        ----------
        v : np.ndarray
            Effect vector of shape (m,).

        Returns
        -------
        np.ndarray
            Result of shape (n,).
        """
        ...

    @abstractmethod
    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Per-SNP standardized diploid matvec.

        Computes ((G - 2p) / sqrt(2p(1-p))) @ v where p is the allele
        frequency vector. Each column is centered AND scaled to unit
        variance under HWE.

        Parameters
        ----------
        v : np.ndarray
            Effect vector of shape (m,).
        af : np.ndarray, optional
            Allele frequencies. If None, recomputed from data.

        Returns
        -------
        np.ndarray
            Result of shape (n,).
        """
        ...

    @abstractmethod
    def recompute_af(self) -> np.ndarray:
        """Recompute empirical allele frequencies from current data.

        Returns
        -------
        np.ndarray
            Allele frequencies of shape (m,).
        """
        ...

    @abstractmethod
    def to_dense(self) -> "DenseHaplotypeArray":
        """Materialize as a DenseHaplotypeArray.

        Returns
        -------
        DenseHaplotypeArray
            Dense representation of the haplotype data.
        """
        ...

    @abstractmethod
    def meiosis(self, assignment, recombination_map) -> "HaplotypeOperator":
        """Perform meiosis to produce offspring haplotypes.

        Parameters
        ----------
        assignment : NMateAssignment
            Mate assignment with maternal/paternal indices and offspring metadata.
        recombination_map : RecombinationMap
            Recombination probabilities between loci.

        Returns
        -------
        HaplotypeOperator
            Offspring haplotypes.
        """
        ...

    @abstractmethod
    def __getitem__(self, key) -> "HaplotypeOperator":
        """Subset by sample/variant indices."""
        ...


class DenseHaplotypeArray(HaplotypeOperator):
    """
    Dense numpy-backed haplotype array implementing both old HaplotypeArray
    interface and new HaplotypeOperator ABC.

    Stores haplotypes as a 3D array with shape (n_samples, m_variants, 2)
    where the last dimension represents the two haplotype copies.
    Convention: genotypes[:,:,0] = maternal, genotypes[:,:,1] = paternal.

    Parameters
    ----------
    genotypes : np.ndarray
        3D array of shape (n, m, 2) containing haplotype data.
    generation : int, optional
        Generation number. Default is 0.
    samples : SampleMeta, optional
        Sample metadata (iid, fid, sex).
    variants : VariantMeta, optional
        Variant metadata (vid, chrom, pos_bp, pos_cM, af, alleles).
    """

    def __init__(
        self,
        genotypes: NDArray[Shape["*, *, *"], Int8],
        generation: int = 0,
        samples: Optional[SampleMeta] = None,
        variants: Optional[VariantMeta] = None,
    ) -> None:
        # call base class init
        self._generation = generation

        # Validate genotypes
        if genotypes is None:
            genotypes = np.empty((0, 0, 2), dtype=np.int8)
        else:
            genotypes = np.asarray(genotypes, dtype=np.int8)

        if genotypes.ndim != 3 or genotypes.shape[2] != 2:
            raise ValueError(
                f"genotypes must be 3-D with last dim = 2, got shape {genotypes.shape}"
            )

        self.genotypes = genotypes
        n, m, _ = genotypes.shape

        # Create default SampleMeta if not provided
        if samples is None:
            iid = np.arange(n, dtype=np.int64)
            self.samples = SampleMeta(iid=iid, generation=generation)
        else:
            if samples.n != n:
                raise ValueError(
                    f"samples.n ({samples.n}) must match genotypes.shape[0] ({n})"
                )
            # Ensure SampleMeta has correct generation
            if samples.generation != generation:
                self.samples = samples.with_generation(generation)
            else:
                self.samples = samples

        # Create default VariantMeta if not provided
        if variants is None:
            vid = np.arange(m, dtype=np.int64)
            self.variants = VariantMeta(vid=vid)
        else:
            if variants.m != m:
                raise ValueError(
                    f"variants.m ({variants.m}) must match genotypes.shape[1] ({m})"
                )
            self.variants = variants

    @property
    def generation(self) -> int:
        """Generation number."""
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    # --- implement base class properties ---

    @property
    def n(self) -> int:
        """Number of samples."""
        return self.genotypes.shape[0]

    @property
    def m(self) -> int:
        """Number of diploid variants."""
        return self.genotypes.shape[1]

    @property
    def diploid_genotypes(self) -> np.ndarray:
        """Return diploid genotype counts (0, 1, or 2) as 2D array (n, m)."""
        return self.genotypes[:, :, 0] + self.genotypes[:, :, 1]

    # --- sample metadata accessors ---

    @property
    def iid(self) -> np.ndarray:
        """Individual IDs."""
        return self.samples.iid

    @property
    def fid(self) -> np.ndarray:
        """Family IDs."""
        return self.samples.fid

    @property
    def sex(self) -> np.ndarray:
        """Biological sex (0=female, 1=male)."""
        return self.samples.sex

    @property
    def n_fam(self) -> int:
        """Number of unique families."""
        return self.samples.n_fam

    @property
    def n_female(self) -> int:
        """Number of biological females."""
        return self.samples.n_female

    @property
    def n_male(self) -> int:
        """Number of biological males."""
        return self.samples.n_male

    # --- variant metadata accessors ---

    @property
    def vid(self) -> np.ndarray:
        """Variant IDs."""
        return self.variants.vid

    @property
    def chrom(self) -> Optional[np.ndarray]:
        """Chromosome for each variant."""
        return self.variants.chrom

    @property
    def pos_bp(self) -> Optional[np.ndarray]:
        """Base pair position."""
        return self.variants.pos_bp

    @property
    def pos_cM(self) -> Optional[np.ndarray]:
        """Centimorgan position."""
        return self.variants.pos_cM

    @property
    def af(self) -> Optional[np.ndarray]:
        """Stored allele frequencies (from VariantMeta)."""
        return self.variants.af

    # --- matrix–vector operations ---

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Diploid matrix-vector product: G @ v where G is the (n, m) dosage matrix."""
        return self.diploid_genotypes @ v

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """Transpose matrix-vector product: G.T @ v."""
        return self.diploid_genotypes.T @ v

    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Per-SNP standardized matvec: ((G - 2p) / sqrt(2p(1-p))) @ v."""
        G_std = self.to_diploid_standardized(af=af, scale=True)
        return G_std @ v

    def diploid_matvec(self, u: np.ndarray) -> np.ndarray:
        """(G[:, :, 0] + G[:, :, 1]) @ u."""
        return (self.genotypes[:, :, 0] @ u) + (self.genotypes[:, :, 1] @ u)

    def standardized_haploid_matvec(self, u: np.ndarray, haploid: int) -> np.ndarray:
        """
        Standardized matvec for one haplotype (0 or 1):
        center & scale each variant column, then multiply by u.
        """
        H = self.genotypes[:, :, haploid]
        col_mean = H.mean(axis=0)
        col_std = H.std(axis=0, ddof=1)
        col_std[col_std == 0] = 1.0  # avoid divide-by-zero
        H_std = (H - col_mean) / col_std
        return H_std @ u

    # --- subsetting ---

    def subset(
        self,
        sample_idx=None,
        variant_idx=None,
        copy: bool = True,
    ) -> "DenseHaplotypeArray":
        """
        Return a new DenseHaplotypeArray with a subset of samples and/or variants.

        Parameters
        ----------
        sample_idx : array-like or slice, optional
            Indices of samples to keep.
        variant_idx : array-like or slice, optional
            Indices of variants to keep.
        copy : bool, optional
            If True, copy the data. Default True.

        Returns
        -------
        DenseHaplotypeArray
            Subsetted haplotype array.
        """
        if sample_idx is None:
            sample_idx = slice(None)
        if variant_idx is None:
            variant_idx = slice(None)

        new_genotypes = self.genotypes[sample_idx, :, :][:, variant_idx, :]
        if copy:
            new_genotypes = new_genotypes.copy()

        new_samples = self.samples.subset(sample_idx)
        new_variants = self.variants.subset(variant_idx)

        return DenseHaplotypeArray(
            genotypes=new_genotypes,
            generation=self.generation,
            samples=new_samples,
            variants=new_variants,
        )

    def __getitem__(self, key) -> "DenseHaplotypeArray":
        """
        Support numpy/xarray-style indexing: haplotypes[sample_idx] or haplotypes[sample_idx, variant_idx].

        Parameters
        ----------
        key : int, slice, array, or tuple
            Index or indices for samples (and optionally variants).

        Returns
        -------
        DenseHaplotypeArray
            Subsetted haplotype array.
        """
        if isinstance(key, tuple):
            if len(key) == 1:
                sample_idx = key[0]
                variant_idx = slice(None)
            elif len(key) == 2:
                sample_idx, variant_idx = key
            else:
                raise IndexError(f"Too many indices: {len(key)}")
        else:
            sample_idx = key
            variant_idx = slice(None)

        return self.subset(sample_idx=sample_idx, variant_idx=variant_idx, copy=False)

    def drop_isel(self, sample=None, variant=None) -> "DenseHaplotypeArray":
        """
        Drop samples or variants by index (xarray-style compatibility).

        Parameters
        ----------
        sample : array-like, optional
            Indices of samples to drop.
        variant : array-like, optional
            Indices of variants to drop.

        Returns
        -------
        DenseHaplotypeArray
            Haplotype array with specified samples/variants removed.
        """
        sample_idx = slice(None)
        variant_idx = slice(None)

        if sample is not None:
            keep_samples = np.ones(self.n, dtype=bool)
            keep_samples[sample] = False
            sample_idx = np.where(keep_samples)[0]

        if variant is not None:
            keep_variants = np.ones(self.m, dtype=bool)
            keep_variants[variant] = False
            variant_idx = np.where(keep_variants)[0]

        return self.subset(sample_idx=sample_idx, variant_idx=variant_idx, copy=True)

    # --- Compatibility methods for xarray-style interface ---

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape as (n_samples, 2*m_variants) for compatibility with 2D expectations."""
        return (self.n, 2 * self.m)

    @property
    def data(self) -> np.ndarray:
        """Return genotypes in 2D interleaved format (n, 2m) for compatibility."""
        # Interleave the two haplotypes: [h0_v0, h1_v0, h0_v1, h1_v1, ...]
        n, m, _ = self.genotypes.shape
        interleaved = np.empty((n, 2 * m), dtype=np.int8)
        interleaved[:, 0::2] = self.genotypes[:, :, 0]
        interleaved[:, 1::2] = self.genotypes[:, :, 1]
        return interleaved

    @property
    def values(self) -> np.ndarray:
        """Alias for data property."""
        return self.data

    @property
    def attrs(self) -> dict:
        """Return attributes dict for compatibility with xarray interface."""
        return {'generation': self.generation}

    @property
    def af_empirical(self) -> np.ndarray:
        """Compute empirical allele frequencies from genotype data."""
        # Mean across samples for each haplotype, then average the two haplotypes
        af_hap0 = self.genotypes[:, :, 0].mean(axis=0)
        af_hap1 = self.genotypes[:, :, 1].mean(axis=0)
        return (af_hap0 + af_hap1) / 2

    def to_diploid_standardized(self, af: np.ndarray = None, scale: bool = False) -> np.ndarray:
        """
        Return standardized diploid genotypes.

        Parameters
        ----------
        af : np.ndarray, optional
            Allele frequencies to use for standardization. If None, uses empirical.
        scale : bool, optional
            If True, scale by sqrt(2*p*(1-p)). Default False.

        Returns
        -------
        np.ndarray
            Standardized diploid genotypes (n, m).
        """
        G = self.diploid_genotypes.astype(np.float64)
        if af is None:
            af = self.af_empirical
        # Center: subtract 2*p (expected value under HWE)
        G_centered = G - 2 * af
        if scale:
            # Scale by sqrt(2*p*(1-p))
            denom = np.sqrt(2 * af * (1 - af))
            denom[denom == 0] = 1.0  # avoid divide by zero
            G_centered = G_centered / denom
        return G_centered

    # --- Deprecated methods for backward compatibility ---

    def get_sample_indexer(self) -> "xft.index.SampleIndex":
        """
        Create a SampleIndex from this haplotype array.

        .. deprecated::
            Use the `samples` attribute directly instead.

        Returns
        -------
        xft.index.SampleIndex
            Sample indexer with data from this array.
        """
        warnings.warn(
            "get_sample_indexer() is deprecated. Use the 'samples' attribute directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return xft.index.SampleIndex(
            iid=self.samples.iid.astype(str),
            fid=self.samples.fid.astype(str),
            sex=self.samples.sex,
            generation=self.generation,
        )

    def get_variant_indexer(self) -> "xft.index.DiploidVariantIndex":
        """
        Create a DiploidVariantIndex from this haplotype array.

        .. deprecated::
            Use the `variants` attribute directly instead.

        Returns
        -------
        xft.index.DiploidVariantIndex
            Variant indexer with data from this array.
        """
        warnings.warn(
            "get_variant_indexer() is deprecated. Use the 'variants' attribute directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return xft.index.DiploidVariantIndex(
            vid=self.variants.vid,
            chrom=self.variants.chrom,
            pos_bp=self.variants.pos_bp,
            pos_cM=self.variants.pos_cM,
            af=self.variants.af if self.variants.af is not None else self.af_empirical,
            zero_allele=self.variants.zero_allele,
            one_allele=self.variants.one_allele,
        )

    # --- HaplotypeOperator ABC implementation ---

    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """Maternal haplotype matvec: hap[:,:,0] @ v."""
        return self.genotypes[:, :, 0] @ v

    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """Paternal haplotype matvec: hap[:,:,1] @ v."""
        return self.genotypes[:, :, 1] @ v

    def recompute_af(self) -> np.ndarray:
        """Recompute and return empirical allele frequencies."""
        return self.af_empirical

    def to_dense(self) -> "DenseHaplotypeArray":
        """Return self (already dense)."""
        return self

    def meiosis(self, assignment, recombination_map) -> "DenseHaplotypeArray":
        """
        Perform meiosis to produce offspring haplotypes.

        Delegates to the existing numba-jitted _meiosis_3d kernel in reproduce.py.

        Parameters
        ----------
        assignment : NMateAssignment
            Mate assignment with maternal/paternal indices and offspring metadata.
        recombination_map : RecombinationMap
            Recombination probabilities between loci.

        Returns
        -------
        DenseHaplotypeArray
            Offspring haplotypes with inherited VariantMeta.
        """
        from xftsim.reproduce import meiosis as _meiosis_fn

        offspring_genotypes = _meiosis_fn(
            self,
            recombination_map,
            assignment.maternal_idx,
            assignment.paternal_idx,
        )

        return DenseHaplotypeArray(
            genotypes=offspring_genotypes,
            generation=assignment.offspring_samples.generation,
            samples=assignment.offspring_samples,
            variants=self.variants,
        )

    @property
    def xft(self) -> "NHaplotypeArrayAccessor":
        """Return accessor object for compatibility with xarray .xft interface."""
        return NHaplotypeArrayAccessor(self)

    def __repr__(self) -> str:
        parts = [f"n={self.n}", f"m={self.m}", f"generation={self.generation}"]
        if self.samples.n_fam != self.n:
            parts.append(f"n_fam={self.n_fam}")
        return f"DenseHaplotypeArray({', '.join(parts)})"


def _extract_variant_meta_from_grg(grg) -> VariantMeta:
    """Extract VariantMeta from a pygrgl GRG object.

    Constructs variant IDs as "{position}:{ref}:{alt}".
    Chromosome information is not available from GRG metadata.
    """
    m = grg.num_mutations
    positions = np.empty(m, dtype=np.int64)
    ref_alleles = np.empty(m, dtype=object)
    alt_alleles = np.empty(m, dtype=object)
    vids = np.empty(m, dtype=object)

    for i in range(m):
        mut = grg.get_mutation_by_id(i)
        positions[i] = int(mut.position)
        ref_alleles[i] = str(mut.ref_allele)
        alt_alleles[i] = str(mut.allele)
        vids[i] = f"{int(mut.position)}:{mut.ref_allele}:{mut.allele}"

    return VariantMeta(
        vid=vids,
        pos_bp=positions,
        zero_allele=ref_alleles,
        one_allele=alt_alleles,
    )


class GraphHaplotypeOperator(HaplotypeOperator):
    """GRG-backed haplotype operator using pygrgl graph traversal.

    Provides O(nodes)-per-variant matvec without materializing the full
    genotype matrix.  After meiosis, offspring revert to DenseHaplotypeArray
    since GRG has no native recombination support.

    Parameters
    ----------
    grg : pygrgl.GRG
        Loaded GRG object (via ``pygrgl.load_immutable_grg``).
    generation : int
        Generation number (default 0).
    samples : SampleMeta, optional
        Sample metadata.  If None, extracted from GRG individual IDs.
    variants : VariantMeta, optional
        Variant metadata.  If None, extracted from GRG mutation data.
    """

    def __init__(self, grg, generation: int = 0,
                 samples: Optional[SampleMeta] = None,
                 variants: Optional[VariantMeta] = None):
        self._grg = grg
        self._generation = generation
        self._cached_af = None

        # --- Samples ---
        if samples is None:
            n = grg.num_individuals
            if grg.has_individual_ids:
                iid = np.array([grg.get_individual_id(i) for i in range(n)])
            else:
                iid = np.arange(n, dtype=np.int64)
            self.samples = SampleMeta(iid=iid, generation=generation)
        else:
            if samples.n != grg.num_individuals:
                raise ValueError(
                    f"samples.n ({samples.n}) != grg.num_individuals ({grg.num_individuals})"
                )
            if samples.generation != generation:
                self.samples = samples.with_generation(generation)
            else:
                self.samples = samples

        # --- Variants ---
        if variants is None:
            self.variants = _extract_variant_meta_from_grg(grg)
        else:
            if variants.m != grg.num_mutations:
                raise ValueError(
                    f"variants.m ({variants.m}) != grg.num_mutations ({grg.num_mutations})"
                )
            self.variants = variants

    # --- properties ---

    @property
    def n(self) -> int:
        return self._grg.num_individuals

    @property
    def m(self) -> int:
        return self._grg.num_mutations

    @property
    def generation(self) -> int:
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    # --- matrix-vector operations ---

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Diploid G @ v via GRG DOWN traversal (by_individual=True)."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=True)  # (K, n)
        result = out.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """G.T @ v via GRG UP traversal (by_individual=True)."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, n)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.UP,
                            by_individual=True)  # (K, m)
        result = out.T  # (m, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """Maternal haplotype matvec via GRG DOWN haploid, even indices."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=False)  # (K, 2n)
        maternal = out[:, 0::2]  # even = maternal
        result = maternal.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """Paternal haplotype matvec via GRG DOWN haploid, odd indices."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=False)  # (K, 2n)
        paternal = out[:, 1::2]  # odd = paternal
        result = paternal.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Per-SNP standardized diploid matvec (no materialization).

        Computes ((G - 2p) / sqrt(2pq)) @ v = (G - 2p) @ (v / sqrt(2pq)).
        """
        v = np.asarray(v, dtype=np.float64)
        if af is None:
            af = self.recompute_af()
        # Scale v by 1/sqrt(2pq) so that (G-2p) @ v_scaled = ((G-2p)/sqrt(2pq)) @ v
        denom = np.sqrt(2.0 * af * (1.0 - af))
        denom[denom == 0] = 1.0
        if v.ndim == 1:
            v_scaled = v / denom
        else:
            v_scaled = v / denom[:, np.newaxis]
        raw = self.matvec(v_scaled)
        correction = 2.0 * (af @ v_scaled)
        return raw - correction

    def recompute_af(self) -> np.ndarray:
        """Compute allele frequencies via GRG UP traversal: G.T @ 1 / (2n)."""
        import pygrgl
        if self._cached_af is not None:
            return self._cached_af
        ones = np.ones((1, self.n), dtype=np.float64)
        counts = pygrgl.matmul(self._grg, ones, pygrgl.TraversalDirection.UP,
                               by_individual=True)  # (1, m)
        self._cached_af = (counts.ravel() / (2.0 * self.n))
        return self._cached_af

    def to_dense(self) -> DenseHaplotypeArray:
        """Materialize full genotype matrix via identity matvec."""
        import pygrgl
        eye = np.eye(self.m, dtype=np.float64)  # (m, m)
        # DOWN haploid: (m, m) -> (m, 2n)
        haploid = pygrgl.matmul(self._grg, eye, pygrgl.TraversalDirection.DOWN,
                                by_individual=False)  # (m, 2n)
        # Reshape to (n, m, 2): maternal=even, paternal=odd
        maternal = haploid[:, 0::2].T  # (n, m)
        paternal = haploid[:, 1::2].T  # (n, m)
        genotypes = np.stack([maternal, paternal], axis=2).astype(np.int8)
        return DenseHaplotypeArray(
            genotypes=genotypes,
            generation=self._generation,
            samples=self.samples,
            variants=self.variants,
        )

    def meiosis(self, assignment, recombination_map) -> DenseHaplotypeArray:
        """Materialize to dense, then perform meiosis."""
        return self.to_dense().meiosis(assignment, recombination_map)

    def __getitem__(self, key) -> DenseHaplotypeArray:
        """Materialize to dense and subset."""
        return self.to_dense()[key]

    def __repr__(self) -> str:
        return (f"GraphHaplotypeOperator(n={self.n}, m={self.m}, "
                f"generation={self.generation})")


class StandardizedHaplotypeOperator(HaplotypeOperator):
    """Wraps a HaplotypeOperator so that ``matvec``/``rmatvec`` act on
    the column-standardized matrix S = (X - mu) / sigma without
    materializing S.

    Identities used (mu, sigma broadcast across rows of X):

    - ``S @ u   = H.matvec(u / sigma) - <mu, u / sigma>``
    - ``S.T @ v = (H.rmatvec(v) - mu * sum(v)) / sigma``

    Defaults follow the HWE convention used elsewhere: ``mu = 2p``,
    ``sigma = sqrt(2 p (1 - p))`` where ``p`` comes from
    ``H.recompute_af()``. Loci with ``sigma == 0`` (monomorphic) keep
    ``sigma = 1`` to avoid division by zero, matching the existing
    ``standardized_matvec`` implementations.

    All other ``HaplotypeOperator`` methods (``matvec_maternal``,
    ``matvec_paternal``, ``recompute_af``, ``to_dense``, ``meiosis``,
    ``__getitem__``) forward to the underlying operator and return
    raw (un-standardized) results. Re-wrap explicitly if you need
    standardized semantics on the result.

    Parameters
    ----------
    haplotypes : HaplotypeOperator
        Underlying operator providing raw genotype matvec.
    means : np.ndarray, optional
        Per-variant means (length m). Defaults to ``2 * af``.
    stds : np.ndarray, optional
        Per-variant standard deviations (length m). Defaults to
        ``sqrt(2 * af * (1 - af))``. Zeros are replaced with 1.
    """

    def __init__(
        self,
        haplotypes: HaplotypeOperator,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ):
        self._H = haplotypes

        if means is None or stds is None:
            af = haplotypes.recompute_af()
            if means is None:
                means = 2.0 * af
            if stds is None:
                stds = np.sqrt(2.0 * af * (1.0 - af))

        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64).copy()
        if means.shape != (haplotypes.m,):
            raise ValueError(
                f"means shape {means.shape} != (m,) = ({haplotypes.m},)"
            )
        if stds.shape != (haplotypes.m,):
            raise ValueError(
                f"stds shape {stds.shape} != (m,) = ({haplotypes.m},)"
            )
        stds[stds == 0] = 1.0

        self._means = means
        self._stds = stds

    # --- forwarded metadata ---

    @property
    def samples(self):
        return self._H.samples

    @property
    def variants(self):
        return self._H.variants

    @property
    def n(self) -> int:
        return self._H.n

    @property
    def m(self) -> int:
        return self._H.m

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    # --- standardized matrix-vector ops ---

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """S @ v = H.matvec(v / sigma) - <mu, v / sigma>."""
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            v_scaled = v / self._stds
            raw = self._H.matvec(v_scaled)
            return raw - self._means @ v_scaled
        v_scaled = v / self._stds[:, np.newaxis]
        raw = self._H.matvec(v_scaled)
        correction = self._means @ v_scaled  # shape (k,)
        return raw - correction[np.newaxis, :]

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """S.T @ v = (H.rmatvec(v) - mu * sum(v)) / sigma."""
        v = np.asarray(v, dtype=np.float64)
        raw = self._H.rmatvec(v)
        if v.ndim == 1:
            return (raw - self._means * v.sum()) / self._stds
        vsum = v.sum(axis=0)  # shape (k,)
        return (raw - self._means[:, np.newaxis] * vsum[np.newaxis, :]) / self._stds[:, np.newaxis]

    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Already standardized: equivalent to ``matvec``.

        The ``af`` argument is ignored; standardization parameters are
        fixed at construction time.
        """
        return self.matvec(v)

    # --- forwarded ops (raw, un-standardized) ---

    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        return self._H.matvec_maternal(v)

    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        return self._H.matvec_paternal(v)

    def recompute_af(self) -> np.ndarray:
        return self._H.recompute_af()

    def to_dense(self) -> "DenseHaplotypeArray":
        return self._H.to_dense()

    def meiosis(self, assignment, recombination_map) -> "HaplotypeOperator":
        return self._H.meiosis(assignment, recombination_map)

    def __getitem__(self, key) -> "HaplotypeOperator":
        return self._H[key]

    def __repr__(self) -> str:
        return (f"StandardizedHaplotypeOperator(H={type(self._H).__name__}, "
                f"n={self.n}, m={self.m})")


class NHaplotypeArrayAccessor:
    """
    Accessor class that mimics the xarray .xft interface for DenseHaplotypeArray.
    Provides compatibility with code expecting xarray-style access.
    """

    def __init__(self, haplotypes: "DenseHaplotypeArray"):
        self._haplotypes = haplotypes

    @property
    def n(self) -> int:
        """Number of samples."""
        return self._haplotypes.n

    @property
    def m(self) -> int:
        """Number of variants."""
        return self._haplotypes.m

    @property
    def generation(self) -> int:
        """Generation number."""
        return self._haplotypes.generation

    @property
    def samples(self) -> SampleMeta:
        """Sample metadata."""
        return self._haplotypes.samples

    @property
    def variants(self) -> VariantMeta:
        """Variant metadata."""
        return self._haplotypes.variants

    @property
    def af_empirical(self) -> np.ndarray:
        """Empirical allele frequencies."""
        return self._haplotypes.af_empirical

    def get_sample_indexer(self) -> "xft.index.SampleIndex":
        """Get sample indexer (deprecated)."""
        return self._haplotypes.get_sample_indexer()

    def get_variant_indexer(self) -> "xft.index.DiploidVariantIndex":
        """Get variant indexer (deprecated)."""
        return self._haplotypes.get_variant_indexer()

    def to_diploid(self) -> np.ndarray:
        """Return diploid genotype counts (0, 1, or 2) as 2D array (n, m)."""
        return self._haplotypes.diploid_genotypes

    def to_diploid_standardized(self, af: np.ndarray = None, scale: bool = False) -> np.ndarray:
        """Return standardized diploid genotypes."""
        return self._haplotypes.to_diploid_standardized(af=af, scale=scale)


# ---------------------------------------------------------------------------
# NPhenotypeArray — new numpy-backed phenotype container
# ---------------------------------------------------------------------------

class NPhenotypeArray:
    """
    Thin wrapper around a flat dict of named 1-D arrays.

    Each key is a component/phenotype name (e.g. 'height.G', 'height').
    The dot is purely a human convention — not parsed.

    Parameters
    ----------
    samples : SampleMeta
        Sample metadata that travels with the data.
    values : dict, optional
        Initial name → (n,) array mapping.
    """

    def __init__(self, samples: SampleMeta, values: Optional[Dict[str, np.ndarray]] = None):
        self.samples = samples
        self._values: Dict[str, np.ndarray] = {}
        if values is not None:
            for k, v in values.items():
                self[k] = v

    def __getitem__(self, key: str) -> np.ndarray:
        return self._values[key]

    def __setitem__(self, key: str, val: np.ndarray):
        val = np.asarray(val, dtype=np.float64)
        if val.shape != (self.samples.n,):
            raise ValueError(
                f"Value for '{key}' has shape {val.shape}, expected ({self.samples.n},)"
            )
        if key in self._values:
            warnings.warn(f"Overwriting existing key '{key}' in NPhenotypeArray")
        self._values[key] = val

    def __contains__(self, key: str) -> bool:
        return key in self._values

    @property
    def keys(self):
        """Return the names of all stored components."""
        return self._values.keys()

    def subset(self, idx) -> "NPhenotypeArray":
        """Return a new NPhenotypeArray with a subset of samples."""
        new_samples = self.samples.subset(idx)
        new_values = {k: v[idx].copy() for k, v in self._values.items()}
        return NPhenotypeArray(samples=new_samples, values=new_values)

    def __repr__(self) -> str:
        return (f"NPhenotypeArray(n={self.samples.n}, "
                f"keys={list(self._values.keys())})")


# ---------------------------------------------------------------------------
# PedigreeArray
# ---------------------------------------------------------------------------

@dataclass
class PedigreeArray:
    """
    Integer index arrays linking offspring to parents.

    Produced at reproduction time; consumed by parent/mother/father references
    and by filters (TrioFilter, SibPairFilter).

    Parameters
    ----------
    offspring_samples : SampleMeta
        Metadata for the offspring generation.
    maternal_idx : np.ndarray
        (n,) indices into the *parent* generation's SampleMeta for each offspring's mother.
    paternal_idx : np.ndarray
        (n,) indices into the *parent* generation's SampleMeta for each offspring's father.
    parent_n : int
        Number of individuals in the parent generation (for bounds checking).
    """
    offspring_samples: SampleMeta
    maternal_idx: np.ndarray
    paternal_idx: np.ndarray
    parent_n: int

    def __post_init__(self):
        self.maternal_idx = np.asarray(self.maternal_idx, dtype=np.intp)
        self.paternal_idx = np.asarray(self.paternal_idx, dtype=np.intp)
        n = self.offspring_samples.n
        if len(self.maternal_idx) != n:
            raise ValueError(
                f"maternal_idx length {len(self.maternal_idx)} != offspring n {n}"
            )
        if len(self.paternal_idx) != n:
            raise ValueError(
                f"paternal_idx length {len(self.paternal_idx)} != offspring n {n}"
            )
        if n > 0:
            if np.any(self.maternal_idx < 0) or np.any(self.maternal_idx >= self.parent_n):
                raise ValueError(
                    f"maternal_idx out of bounds [0, {self.parent_n})"
                )
            if np.any(self.paternal_idx < 0) or np.any(self.paternal_idx >= self.parent_n):
                raise ValueError(
                    f"paternal_idx out of bounds [0, {self.parent_n})"
                )