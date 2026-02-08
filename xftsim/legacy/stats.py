import xftsim as xft
import warnings
import functools
import numpy as np
import numba as nb
import pandas as pd
import nptyping as npt
import scipy as sp
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property
import dask.array as da



class Statistic:
    """
    Base class for defining statistic estimators.

    Attributes
    ----------
    name : str
        The name of the statistic.
    estimator : Callable
        The function that estimates the statistic.
    metadata : Dict
        Any additional metadata
    filter_sample : xft.filters.SampleFilter
        Apply global filter prior to estimation

    Methods
    -------
    estimate(sim: xft.sim.Simulation) -> None:
        Estimate the statistic and update the results.

    update_results(sim: xft.sim.Simulation, results: object) -> None:
        Update the simulation's results_store with the estimated results.
    """
    def __init__(self, 
                 estimator: Callable,
                 parser: Callable,
                 name: str = None,
                 metadata: Dict = {},
                 sample_filter = xft.filters.PassFilter(),
                 s_args: Iterable = None,
                 ):
        self.name = name
        self.estimator = estimator
        self.parser = parser
        self.sample_filter = sample_filter
        self.s_args = s_args

    def update_results(self, sim: xft.sim.Simulation, results: object) -> None:
        ## initialize empty dict if result store for generation is empty
        if sim.generation not in sim.results_store.keys():
            sim.results_store[sim.generation] = {}
        ## initialize empty dict if estimator hasn't been used before in current generation
        if self.name not in sim.results_store[sim.generation].keys():
            sim.results_store[sim.generation][self.name] = {}
        ## append results to simulation results_store
        sim.results_store[sim.generation][self.name] = results

    def parse_results(self, sim: xft.sim.Simulation) -> None:
        if 'parsed' not in sim.results_store.keys():
            sim.results_store['parsed'] = {}
        results = sim.results_store[sim.generation][self.name]
        parsed = self.parser(sim, results)
        if parsed is None:
            pass
        else:
            sim.results_store['parsed'].update(parsed)
    
    def estimate(self,
                 sim: xft.sim.Simulation = None,
                 **kwargs) -> None:
        ## The goal here is that specific kwargs can take precidence over the sim object when specified
        ## For example, if you provide a subsample of genotypes/phenotypes, the estimator will be applied to
        ## those, otherwise it will retrieve them from the sim object
        if self.s_args is None:
            output = self.estimator(sim)
        else:
            inputs = dict()
            for arg in self.s_args:
                if arg in kwargs.keys():
                    inputs[arg] = kwargs[arg]
                else:
                    inputs[arg] = getattr(sim, arg) 
            output = self.estimator(**inputs)
        self.update_results(sim, output)

    @staticmethod
    def null_parser(self, *args, **kwargs):
        pass


class SampleStatistics(Statistic):
    """
    Calculate and return various sample statistics for the given simulation.

    Attributes
    ----------
    means : bool
        If True, calculate and return the mean of each phenotype.
    variance_components : bool
        If True, calculate and return the variance components of each phenotype. Overidden to FALSE if
        component_index is `pheno`
    variances : bool
        If True, calculate and return the variances of each phenotype.
    vcov : bool
        If True, calculate and return the variance-covariance matrix.
    corr : bool
        If True, calculate and return the correlation matrix.
    prettify : bool
        If True, prettify the output by converting it to a pandas DataFrame.
    component_index: str, xft.index.ComponentIndex
        If `all`, will compute statistics for all phenotype components
        If `pheno`, will compute statistics for phenotypes only
        If a ComponentIndex object, will compute statistics for specified components only

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Calculate and return the requested sample statistics for the given simulation.
    """
    def __init__(self,
                 means: bool = True,
                 variance_components: bool = True,
                 variances: bool = True,
                 vcov: bool = True,
                 corr: bool = True,
                 prettify: bool = True,
                 metadata: Dict = {},
                 sample_filter = xft.filters.PassFilter(),
                 name: str = 'sample_statistics',
                 component_index: Union[str,xft.index.ComponentIndex] = 'all',
                 ):
        self.name = name
        self.means = means
        self.variances = variances
        self.variance_components = variance_components
        if not isinstance(component_index, xft.index.ComponentIndex):
            if component_index == 'pheno':
                self.variance_components = False
        self.vcov = vcov
        self.corr = corr
        self.prettify = prettify
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.s_args = ['phenotypes']
        self.parser = Statistic.null_parser
        self.component_index = component_index

    @xft.utils.profiled(level=2, message = "sample statistics")
    def estimator(self, phenotypes: xr.DataArray) -> Dict:
        if isinstance(self.component_index, xft.index.ComponentIndex):
            component_index=self.component_index
        elif self.component_index == 'pheno':
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        elif self.component_index == 'all':
            component_index = phenotypes.xft.get_component_indexer()
        else:
            raise RuntimeError('Invalid component indexer')
        output = dict()
        sind = self.sample_filter.filter(phenotypes)
        pheno_df = phenotypes.xft[None, component_index].xft.as_pd(prettify=self.prettify).iloc[sind,:]
        h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
        mm = [g.mean(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
        if self.means:
            output['means'] = pd.concat([g for g in mm])
        if self.variances:
            output['variances'] = pd.concat([g for g in h])
        if self.variance_components:
            output['variance_components'] = pd.concat([x.iloc[:-1] for x in [g/g.iloc[g.index.get_locs([slice(None),'phenotype'])].values for g in h if g.size>1] if x.size>1])
        if self.vcov:
            output['vcov'] = pheno_df.cov()
        if self.corr:
            output['corr'] = pheno_df.corr()
        return output


class MatingStatistics(Statistic):
    """
    Calculate and return various mating statistics for the given simulation.

    Parameters
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
    full: bool
        Ignore component_index and compute statistics for all components
        If component_index is not provided, and full = False, calculate statistics for phenotype components only.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Calculate and return the requested mating statistics for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 full: bool = False,
                 metadata: Dict = {},
                 sample_filter = xft.filters.PassFilter(),
                 name: str = 'mating_statistics',
                 ):
        self.name = name
        self._full = full
        self.component_index = component_index
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.s_args = ['mating','phenotypes']
        self.parser = Statistic.null_parser

    @xft.utils.profiled(level=2, message = "mating statistics")
    def estimator(self, 
                  phenotypes: xr.DataArray,
                  mating: xft.mate.MateAssignment) -> Dict:
        # if self.filter_sample:
        #     phenotypes = sim.phenotypes_filtered         
        # else:
        #     phenotypes = sim.phenotypes
        output = dict(
                      n_reproducing_pairs = mating.n_reproducing_pairs,
                      n_total_offspring = mating.n_total_offspring,
                      mean_n_offspring_per_pair =  np.mean(mating.n_offspring_per_pair),
                      mean_n_female_offspring_per_pair =  np.mean(mating.n_females_per_pair),
                      mate_correlations = mating.get_mate_phenotypes(phenotypes=phenotypes,
                                                                         component_index=self.component_index,
                                                                         full = self._full).corr(),
                      )
        return output

    

# ## estimates tr (X %*% t(X) %*% X %*% t(X)) using l probing vectors
@nb.njit(parallel=True)
def _rtrace_K2(X, l): ## (matrix, int) -> float
    """
    Estimate the trace of the matrix product (X %*% t(X) %*% X %*% t(X)) using l probing vectors.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    float
        The estimated trace of the matrix product.
    """
    probes = np.random.randn(X.shape[0], l)
    W = probes.T @ (X @ (X.T @ (X @ (X.T @ probes))))
    return np.trace(W)/l

def _rtrace_K2_dask(X, l): ## (matrix, int) -> float
    """
    Estimate the trace of the matrix product (X %*% t(X) %*% X %*% t(X)) using l probing vectors with Dask.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    float
        The estimated trace of the matrix product.
    """
    probes = np.random.randn(X.shape[0], l)
    W = da.dot(probes.T, da.dot(X, da.dot(X.T, da.dot(X, da.dot(X.T, probes)))))
    return da.trace(W)/l

@nb.njit(parallel=True)
def _haseman_elston_randomized(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    l, # (int) number of random probes for trace estimation
                    ):
    """
    Perform randomized Haseman-Elston regression.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    n,m = G.shape
    Ky = G @ (G.T @ Y)/m
    trK2 = _rtrace_K2(G/np.sqrt(m), l)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = (Y.T @ (Ky - Y)) / denom
    return cov_g_HE

def _haseman_elston_randomized_dask(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    l, # (int) number of random probes for trace estimation
                    ):
    """
    Perform randomized Haseman-Elston regression with Dask.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    l : int
        The number of random probes for trace estimation.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    n,m = G.shape
    Ky = da.dot(G, da.dot(G.T, Y))/m
    trK2 = _rtrace_K2_dask(G/np.sqrt(m), l)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = da.dot(Y.T, (Ky - Y)) / denom
    return cov_g_HE.compute()


@nb.njit(parallel=True)
def _haseman_elston_deterministic(
                    G, # (2D array) standardized (but not scaled) diploid genotypes
                    Y, # (2D array) standardized phenotypes
                    ):
    """
    Perform deterministic Haseman-Elston regression.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    n,m = G.shape
    Ky = G @ (G.T @ Y)/m
    trK2 = np.trace(G @ (G.T @ G @ (G.T/m))/m)
    denom = trK2 - n

    ## estimate genetic covariances
    cov_g_HE = (Y.T @ (Ky - Y)) / denom
    return cov_g_HE



def haseman_elston(
                   G, # (2D array) standardized (but not scaled) diploid genotypes
                   Y, # (2D array) standardized phenotypes
                   n_probe = 500, # (int) number of random probes for trace estimator, inf -> deterministic
                   dtype=np.float32,
                   dask: bool = False,
                   ):
    """
    Perform Haseman-Elston regression, with the option to choose randomized, deterministic, or randomized dask-based methods.

    Parameters
    ----------
    G : np.ndarray
        A 2D numpy array representing standardized (but not scaled) diploid genotypes.
    Y : np.ndarray
        A 2D numpy array representing standardized phenotypes.
    n_probe : int, optional, default=500
        The number of random probes for trace estimation. If n_probe is set to inf, use deterministic method.
    dtype : numpy data type, optional, default=np.float32
        The data type for the input arrays.
    dask : bool, optional, default=False
        If True, use dask for calculations.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the estimated genetic covariances.
    """
    Y = xft.utils.ensure2D(Y).astype(dtype)
    G = np.array(G)#, dtype=dtype)
    if np.isinf(n_probe):
        return _haseman_elston_deterministic(G, Y)
    elif dask:
        return _haseman_elston_randomized_dask(G, Y, n_probe)
    else:
        return _haseman_elston_randomized(G, Y, n_probe)

  
class HasemanElstonEstimator(Statistic):
    """
    Estimate Haseman-Elston regression for the given simulation.

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all components.
    genetic_correlation : bool
        If True, calculate and return the genetic correlation matrix.
    randomized : bool
        If True, use a randomized trace estimator.
    prettify : bool
        If True, prettify the output by converting it to a pandas DataFrame.
    n_probe : int
        The number of random probes for trace estimation.
    dask : bool
        If True, use dask for calculations.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Estimate and return the Haseman-Elston regression for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 genetic_correlation: bool = True,
                 randomized: bool = True,
                 prettify: bool = True,
                 n_probe: int = 100,
                 dask: bool = True,
                 metadata: Dict = {},
                 sample_filter = xft.filters.PassFilter(),
                 name: str = 'HE_regression',
                 ):
        self.name = name
        self.component_index = component_index
        self.genetic_correlation = genetic_correlation
        self.randomized = randomized
        self.prettify = prettify
        self.n_probe = n_probe
        self.dask = dask
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.s_args = ['phenotypes', 'haplotypes']
        self.parser = Statistic.null_parser

    @xft.utils.profiled(level=2, message = "haseman elston estimator")
    def estimator(self, 
                  phenotypes,
                  haplotypes) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index

        subinds = self.sample_filter.filter(phenotypes)
        Y = phenotypes.xft[None, component_index][subinds,:].xft.standardize().xft.as_pd()
        G = haplotypes[subinds,:].xft.to_diploid_standardized()
        he_out = haseman_elston(G = G,
                                Y = Y,
                                n_probe = self.n_probe,
                                dask = self.dask,
                                )
        output = dict()
        output['cov_HE'] = pd.DataFrame(he_out, index = Y.columns, columns = Y.columns)
        if self.genetic_correlation:
            output['corr_HE'] = xft.utils.cov2cor(output['cov_HE'])
        return output




  
class HasemanElstonEstimatorSibship(Statistic):
    """
    Estimate Haseman-Elston regression for the given simulation.

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all components.
    genetic_correlation : bool
        If True, calculate and return the genetic correlation matrix.
    randomized : bool
        If True, use a randomized trace estimator.
    prettify : bool
        If True, prettify the output by converting it to a pandas DataFrame.
    n_probe : int
        The number of random probes for trace estimation.
    dask : bool
        If True, use dask for calculations.

    Methods
    -------
    estimator(sim: xft.sim.Simulation) -> Dict
        Estimate and return the Haseman-Elston regression for the given simulation.
    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 genetic_correlation: bool = True,
                 randomized: bool = True,
                 prettify: bool = True,
                 n_probe: int = 100,
                 dask: bool = True,
                 metadata: Dict = {},
                 sample_filter = xft.filters.SibpairSampleFilter(),
                 name: str = 'HE_regression_sibship',
                 ):
        self.name = name
        self.component_index = component_index
        self.genetic_correlation = genetic_correlation
        self.randomized = randomized
        self.prettify = prettify
        self.n_probe = n_probe
        self.dask = dask
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.s_args = ['phenotypes', 'haplotypes']
        self.parser = Statistic.null_parser

    @xft.utils.profiled(level=2, message = "haseman elston estimator sibship")
    def estimator(self, 
                  phenotypes,
                  haplotypes,
                  ) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        # if self.filter_sample:
            # Y = sim.current_std_phenotypes_filtered.xft[None, component_index].xft.as_pd()
            # G = sim.current_std_genotypes_filtered
        # else:

        subinds = self.sample_filter.filter(phenotypes)
        Y0 = phenotypes.xft[None, component_index][subinds,:]
        Y = Y0.xft.standardize().xft.as_pd()
        G0 = haplotypes[subinds,:]

        Y = Y.iloc[0::2,:]-Y.iloc[1::2,:].values
        ys=xft.utils.standardize_array(Y)
        for i in range(ys.shape[1]):
            Y.iloc[:,i] = ys[:,i]
        G = G0.xft.to_diploid_standardized()
        G = xft.utils.standardize_array(G[0::2,:]-G[1::2,:])
        he_out = haseman_elston(G = G,
                                Y = Y,
                                n_probe = self.n_probe,
                                dask = self.dask,
                                )
        output = dict()
        output['cov_HE'] = pd.DataFrame(he_out, index = Y.columns, columns = Y.columns)
        if self.genetic_correlation:
            output['corr_HE'] = xft.utils.cov2cor(output['cov_HE'])
        output['meta'] = dict(G_shape=G0.shape,
                              G_ind=G0.xft.get_sample_indexer(),
                              Y_shape=Y0.shape,
                              Y_ind=Y0.xft.get_sample_indexer(),
                              Y_mean=Y0.to_pandas().mean(),
                              Y_corr=Y0.to_pandas().corr(),
                              Ydiff_mean=Y.mean(),
                              Ydiff_corr=Y.corr(),
                              ) 
        return output






        # h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype


# def _average_sib_corr(a: NDArray):
#     ac = np.corrcoef(a)
#     acshape = ac.shape
#     if acshape[0] == 1:
#         return np.NaN
#     else:
#         return np.mean(ac[np.tril_indices(ac.shape[0])])


# class SiblingSampleStatistics(Statistic):
#     def __init__(self,
#                  variance_components: bool = True,
#                  variances: bool = True,
#                  vcov: bool = True,
#                  corr: bool = True,
#                  prettify: bool = True,
#                  ):
#         self.name = 'sample_statistics'
#         self.variances = variances
#         self.variance_components = variance_components
#         self.vcov = vcov
#         self.corr = corr
#         self.prettify = prettify

#     def processor(self, sim: xft.sim.Simulation) -> Dict:
#         output = dict()
#         pheno_df = sim.phenotypes.xft.as_pd(prettify=self.prettify)
#         h = [g.var(axis=1) for _, g in pheno_df.T.groupby(['phenotype_name','vorigin_relative'])]
#         if self.variances:
#             output['variances'] = pd.concat([g for g in h])
#         if self.variance_components:
#             output['variance_components'] = pd.concat([x.iloc[:-1] for x in [g/g.iloc[g.index.get_locs([slice(None),'phenotype'])].values for g in h] if x.size>1])
#         if self.vcov:
#             output['vcov'] = pheno_df.cov()
#         if self.corr:
#             output['corr'] = pheno_df.corr()
#         return output


@nb.njit
def _linear_regression_with_intercept_nb(x: NDArray,
                                         y: NDArray,
                                         ) -> NDArray:
    """
    Numba implementation of simple linear regression of y ~ 1 + X. 
    Intercepts term is included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-1 array of predictor
    y : NDArray
        n-by-1 standardized outcome array
    
    Returns
    -------
    NDArray
        2 array consisting of standardized slope corresponding standard errors.
    """
    n = x.shape[0]
    pred = np.hstack((np.ones_like(x), x))
    XtX = pred.T @ pred
    XtXinv = np.linalg.pinv(XtX)
    coef = XtXinv @ (pred.T @ y)
    yhat = pred @ coef
    res = y - yhat
    s2 = (res.T @ res) / (n - 2)
    std_err = np.sqrt(s2 * np.diag(XtXinv))
    return np.array([coef.ravel()[1], std_err.ravel()[1]], dtype =np.float32)


@nb.njit
def _mv_vec_linear_regression_with_intercept_nb(X: NDArray,
                                                Y: NDArray,
                                                std_X: bool = True,
                                                std_Y: bool = True,
                                                output_dtype=np.float64) -> NDArray:
    """
    Numba implementation of simple linear regression vectorized over predictors
    and outcomes. 
    Intercept terms are included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    Y : NDArray
        n-by-k standardized outcome arrays
    
    Returns
    -------
    NDArray
        m-by-2-by-k array of standardized slopes (first column) and corresponding standard errors (second column)
        for each of the m predictors. 
    """
    # output = np.empty((X.shape[1], 2, Y.shape[1]), dtype=Y.dtype)
    # for k in nb.prange(Y.shape[1]):
    #     y = Y[:,k].ravel()
    #     y = y / np.std(y)
    #     for j in np.arange(X.shape[1]):
    #         output[j,:, k] = _linear_regression_with_intercept_nb(X[:,j],y)
    # return output
    output = np.empty((X.shape[1], 2, Y.shape[1]), dtype=output_dtype)
    for k in nb.prange(Y.shape[1]):
        y = Y[:,k].ravel()
        y = y.reshape((y.shape[0], 1))
        if std_Y:
            y = y / np.std(y)
        for j in range(X.shape[1]):
            x = X[:,j].ravel()
            x = x.reshape((x.shape[0], 1))
            if std_X:
                x = x/ np.std(x)
            output[j,:, k] = _linear_regression_with_intercept_nb(x = x,
                                                                  y = y, 
                                                                  )
    return output



def _p_val_t(t: float, df: Union[int, float]) -> float:
    """
    p-values for student's t distribution
    
    Parameters
    ----------
    t : float
        Test statistic
    df : int or float
        degrees of freedom
    
    Returns
    -------
    float
        p-value
    """
    return 2* sp.stats.t.cdf(-np.abs(t),df)


def _mv_gwas_nb(X: NDArray, 
                Y: NDArray, 
                std_X: bool,
                std_Y: bool,
                ) -> NDArray:

    """
    Numba implementation of simple linear regression vectorized over predictors and outcomes 
    Intercepts term is included but corresponing coefficients are not reported.
    Both X and Y columns are automatically scaled to have unit variance.
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    Y : NDArray
        n-by-k outcome array
    std_X : bool
        Should columns of X be standardized?
    std_Y : bool
        Should columns of Y be standardized?
    
    Returns
    -------
    NDArray
        m-by-4-by-k array with columns respectively corresponding to
        slopes, standard errors, test statistics, and p-values 
        for each of the m predictors. 
    """
    tmp = _mv_vec_linear_regression_with_intercept_nb(X=X,
                                                      Y=Y,
                                                      std_X=std_X,
                                                      std_Y=std_Y)
    output = np.tile(tmp,(1,2,1))
    output[:,2,:] = output[:,0,:] /output[:,1,:] 
    output[:,3,:] = _p_val_t(output[:,2,:], X.shape[0]-2) 
    return(output)




class GWAS_Estimator(Statistic):
    """

    BROKEN
    Perform linear assocation studies for the given simulation.

    When called within a Simulation, will add to Simulation.results['GWAS']
    a 3-D array indexed as follows:
        - the first dimension indexes variants via xft.index.DiploidVariantIndex
        - the second dimension indexes four association statistics: slope, se, test-statistic, and p-value
        - the third dimension indexes phenotypic components via xft.index.ComponentIndex

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all phenotype components.

    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 metadata: Dict = {},
                 sample_filter = xft.filters.PassFilter(),
                 std_X: bool = True,
                 std_Y: bool = True,
                 name: str = 'GWAS',
                 ):
        self.name = name
        self.component_index = component_index
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.std_X = std_X
        self.std_Y = std_Y
        self.s_args = ['phenotypes', 'haplotypes']
        self.parser = Statistic.null_parser
    # self.numba = numba

    @xft.utils.profiled(level=2, message = "GWAS estimator")
    def estimator(self, phenotypes,
                  haplotypes,
                  ) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        # if self.filter_sample:
            # Y = sim.current_std_phenotypes.xft[None, component_index].data.astype(np.float32)
            # G = sim.current_std_genotypes
        # else:

        subinds = self.sample_filter.filter(phenotypes)

        Y = phenotypes.xft[None, component_index][subinds,:].xft.standardize()

        G = haplotypes[subinds,:].xft.to_diploid_standardized()

        sum_stats = _mv_gwas_nb(G,Y, std_X=self.std_X, std_Y=self.std_Y)
        coord_dict = component_index.coord_dict.copy()
        coord_dict.update(haplotypes.variants.to_variant_index(af=haplotypes.af_empirical).coord_dict)
        if self.std_X and self.std_Y:
            coord_dict.update({'statistic':('statistic', ['std_beta', 'se', 't', 'p'])})
        else:
            coord_dict.update({'statistic':('statistic', ['beta', 'se', 't', 'p'])})
        output = dict(estimates=xr.DataArray(sum_stats,dims=('variant', 'statistic', 'component'), 
                                        coords=coord_dict))
        return output




@nb.njit
def _vec_linear_regression_with_intercept_nb(X: NDArray,
                                             y: NDArray) -> NDArray:
    """
    Numba implementation of simple linear regression vectorized over predictors. 
    Intercepts term is included but corresponing coefficients are not reported
    
    Parameters
    ----------
    X : NDArray
        n-by-m array of predictors
    y : NDArray
        n-by-1 standardized outcome array
    
    Returns
    -------
    NDArray
        m-by-2 array of standardized slopes (first column) and corresponding standard errors (second column)
        for each of the m predictors. 
    """
    output = np.empty((X.shape[1], 2), dtype=y.dtype)
    for j in np.arange(X.shape[1]):
        output[j,:] = _linear_regression_with_intercept_nb(X[:,j],y)
    return output


def threshold_PGS(estimates, threshold, G):
    mask = estimates.loc[:,'p'].values<threshold
    out = G[:,mask] @ estimates.loc[:,'beta'][mask]
    return out.assign_coords({'threshold':threshold,'n_sig':np.sum(mask)})

def apply_threshold_PGS(estimates, G, thresholds = 10**np.linspace(np.log10(5e-8), 0,24)):
    output = [threshold_PGS(estimates, threshold, G) for threshold in thresholds]
    return xr.concat(output,'threshold')

def apply_threshold_PGS_all(gwas_results, G, minp=5e-8, maxp=1, nthresh=25, alpha=.05,
                            thresholds = None):
    if thresholds is None:
        bonferoni = alpha/gwas_results.shape[0]
        if nthresh >= 2:
            thresholds = 10**np.linspace(np.log10(minp),np.log10(maxp),nthresh)
            thresholds = np.sort(np.concatenate([thresholds,[.05,bonferoni]]))
        else:
            thresholds = [alpha,bonferoni]
    output = [apply_threshold_PGS(gwas_results.loc[:,:,comp], G,thresholds) for comp in gwas_results.component.values]
    return xr.concat(output,'component')

    

class Pop_GWAS_Estimator(Statistic):
    """
    Perform one sib only linear assocation studies for the given simulation.

    NOTE! Currently assumes each mate-pair produces exactly 2 offspring

    When called within a Simulation, will add to Simulation.results['GWAS']
    a 3-D array indexed as follows:
        - the first dimension indexes variants via xft.index.DiploidVariantIndex
        - the second dimension indexes four association statistics: slope, se, test-statistic, and p-value
        - the third dimension indexes phenotypic components via xft.index.ComponentIndex

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all phenotype components.

    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 metadata: Dict = {},
                 std_X: bool = False,
                 std_Y: bool = False,
                 assume_pairs: bool = True,
                 n_sub: int = 0,
                 PGS: bool = True,
                 PGS_sub_divisions: int = 50,
                 training_fraction: float = .8,
                 name: str = 'pop_GWAS',
                 sample_filter=xft.filters.UnrelatedSampleFilter(),
                 ):
        self.name = name
        self.component_index = component_index
        self.metadata = metadata
        self.std_X = std_X
        self.std_Y = std_Y
        self.n_sub = n_sub
        self.PGS = PGS
        self.PGS_sub_divisions =PGS_sub_divisions
        self.s_args = ['phenotypes', 'haplotypes']
        self.parser = Statistic.null_parser
        self.training_fraction = training_fraction
        self.sample_filter = sample_filter

        if not assume_pairs:
            raise NotImplementedError()
        else:
            self.assume_pairs = assume_pairs

    @xft.utils.profiled(level=2, message = "pop GWAS estimator")
    def estimator(self, 
                  phenotypes,
                  haplotypes,
                  ) -> Dict:
        if self.component_index is None:
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        n_sib = int(np.floor(phenotypes.shape[0]//2))
        # n_sub = np.min([self.sample_filter.nsub, n_sib])
        n_sub = self.sample_filter.nsub
        NSUB = int(np.floor(self.training_fraction*n_sub))
        self.sample_filter.nsub=NSUB

        subinds = self.sample_filter.filter(phenotypes)

        Y = phenotypes.xft[None, component_index].data.astype(np.float32)[subinds, :]
        Y = np.ascontiguousarray(Y, dtype = np.float32)
        G = haplotypes.data[subinds,0::2] + haplotypes.data[subinds,1::2]
        G = np.ascontiguousarray(G, dtype = np.float32)

        sum_stats = _mv_gwas_nb(G,Y, std_X=self.std_X, std_Y=self.std_Y)
        coord_dict = component_index.coord_dict.copy()
        coord_dict.update(haplotypes.variants.to_variant_index(af=haplotypes.af_empirical).coord_dict)
        if self.std_X and self.std_Y:
            coord_dict.update({'statistic':('statistic', ['std_beta', 'se', 't', 'p'])})
        else:
            coord_dict.update({'statistic':('statistic', ['beta', 'se', 't', 'p'])})
        estimates=xr.DataArray(sum_stats,dims=('variant', 'statistic', 'component'), 
                               coords=coord_dict)
        if self.PGS:
            G = haplotypes.drop_isel(sample=subinds).xft.to_diploid()
            b = estimates.loc[:,'beta',:]   
            PGS = dict(scores=apply_threshold_PGS_all(estimates, G,nthresh=self.PGS_sub_divisions),
                     phenotypes=phenotypes.drop_isel(sample=subinds))
        else:
            PGS = None
        output = dict(estimates=estimates,
                      PGS=PGS,
                      info=dict(n_sub=n_sub,
                                std_X=self.std_X,
                                std_Y=self.std_Y,
                                training_samples = phenotypes.xft.get_sample_indexer().frame.iloc[subinds,],
                                samples = phenotypes.xft.get_sample_indexer().frame,
                                ))
        return output
    # @xft.utils.profiled(level=2, message = "pop GWAS estimator parsing")
    # def parser(self, sim: xft.sim.Simulation, results: dict):
    #     ## GWAS estimates
    #     estimates = results['estimates']
    #     beta_true = sim.bet


        # phenotypes = sim.phenotypes.sel(sample=estimates.)



class Sib_GWAS_Estimator(Statistic):
    """
    Perform sib-difference linear assocation studies for the given simulation.

    NOTE! Currently assumes each mate-pair produces exactly 2 offspring

    When called within a Simulation, will add to Simulation.results['GWAS']
    a 3-D array indexed as follows:
        - the first dimension indexes variants via xft.index.DiploidVariantIndex
        - the second dimension indexes four association statistics: slope, se, test-statistic, and p-value
        - the third dimension indexes phenotypic components via xft.index.ComponentIndex

    Attributes
    ----------
    component_index : xft.index.ComponentIndex, optional
        Index of the component for which the statistics are calculated.
        If not provided, calculate statistics for all phenotype components.

    """
    def __init__(self,
                 component_index: xft.index.ComponentIndex = None,
                 metadata: Dict = {},
                 std_X: bool = False,
                 std_Y: bool = False,
                 assume_pairs: bool = True,
                 n_sub: int = 0,
                 PGS:bool = True,
                 PGS_sub_divisions: int = 50,
                 training_fraction: float = .8,
                 name: str = 'sib_GWAS',
                 sample_filter = xft.filters.SibpairSampleFilter(),
                 ):
        self.name = name
        self.component_index = component_index
        self.metadata = metadata
        self.sample_filter = sample_filter
        self.std_X = std_X
        self.std_Y = std_Y
        self.n_sub = n_sub
        self.PGS=PGS
        self.PGS_sub_divisions =PGS_sub_divisions
        self.parser = Statistic.null_parser
        self.training_fraction = training_fraction
        if not assume_pairs:
            raise NotImplementedError()
        else:
            self.assume_pairs = assume_pairs
        self.s_args = ['phenotypes', 'current_std_phenotypes', 'current_std_genotypes', 'haplotypes']
    @xft.utils.profiled(level=2, message = "sib GWAS estimator")
    def estimator(self, 
                  phenotypes,
                  current_std_phenotypes,
                  current_std_genotypes,
                  haplotypes,
                  ) -> Dict:
        ## look for "phenotype" components if component_index not provided
        if self.component_index is None:
            # pheno_cols= sim.phenotypes.component_name.values[sim.phenotypes.component_name.str.contains('phenotype')]
            # component_index = sim.phenotypes.xft.get_component_indexer()[dict(component_name=pheno_cols)]
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        n_sib = int(np.floor(phenotypes.shape[0]//2))
        n_sub = int(np.floor(self.n_sub))

        if n_sub == 0:
            n_sub = n_sib
        else:
            n_sub = n_sib
        # NSUB = int(np.floor(self.training_fraction*n_sub))
        # self.sample_filter.nsub=NSUB
        n_sub = self.sample_filter.nsub
        NSUB = int(np.floor(self.training_fraction*n_sub))
        self.sample_filter.nsub=NSUB
        train_inds = self.sample_filter.filter(phenotypes)

        Y = phenotypes.xft[None, component_index].data.astype(np.float32)[train_inds,:]
        Y = Y[0::2,:] - Y[1::2,:]
        Y = np.ascontiguousarray(Y, dtype = np.float32)
        G_train = haplotypes.data[train_inds,0::2] + haplotypes.data[train_inds,1::2]
        G_train = G_train[0::2,:] - G_train[1::2,:]
        G_train = np.ascontiguousarray(G_train, dtype = np.float32)

        sum_stats = _mv_gwas_nb(G_train,Y, std_X=self.std_X, std_Y=self.std_Y)
        coord_dict = component_index.coord_dict.copy()
        coord_dict.update(haplotypes.variants.to_variant_index(af=haplotypes.af_empirical).coord_dict)
        # if self.std_X and self.std_Y:
            # coord_dict.update({'statistic':('statistic', ['beta', 'se', 't', 'p'])})
        # else:
        coord_dict.update({'statistic':('statistic', ['beta', 'se', 't', 'p'])})
        estimates=xr.DataArray(sum_stats,dims=('variant', 'statistic', 'component'), 
                               coords=coord_dict)
        if self.PGS:
            if self.std_X:
                raise NotImplementedError()
            else:
                G = haplotypes.drop_isel(sample=train_inds).xft.to_diploid()
                b = estimates.loc[:,'beta',:]   
                PGS = dict(scores=apply_threshold_PGS_all(estimates, G,nthresh=self.PGS_sub_divisions),
                           phenotypes=phenotypes.drop_isel(sample=train_inds))
        else:
            PGS = None
        output = dict(estimates=estimates,
                      PGS=PGS,
                      info=dict(n_sub=n_sub,
                                std_X=self.std_X,
                                std_Y=self.std_Y,
                                training_samples = phenotypes.xft.get_sample_indexer().frame.iloc[train_inds,],
                                ))
        return output


