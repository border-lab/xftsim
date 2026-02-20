import sys
import warnings

import numpy as np

if sys.version_info < (3, 12):
    warnings.warn(
        f"xftsim recommends Python >= 3.12 (you have {sys.version_info.major}.{sys.version_info.minor}). "
        "Some features may not work correctly on older versions.",
        stacklevel=2,
    )

__version__="0.3.0.dev106"


class Config:
    """
    A class to store configuration settings. Instantiated as xftsim.config when package is loaded

    Attributes
    ----------
    nthreads : int
        Number of threads to use for parallel execution.
    print_level : int
        Verbosity level for print statements.
    print_durations_threshold : float
        Threshold for printing durations.
    """

    def __init__(self):
        """
        Initialize the Config object with default settings.
        """
        self.nthreads = 1
        self.print_level = 2
        self.print_durations_threshold = 0. #np.inf

    def get_pdurations(self):
        """
        Get the current print durations threshold.

        Returns
        -------
        float
            The print durations threshold.
        """
        return self.print_durations_threshold

    def get_plevel(self):
        """
        Get the current print level.

        Returns
        -------
        int
            The print level.
        """
        return self.print_level


config = Config()

from . import utils       ## utility functions
from . import data        ## download recombination maps etc (legacy shim)
from . import index       ## indexing (legacy shim)
from . import struct      ## data structures
from . import effect      ## genetic effects (legacy shim)
from . import arch        ## phenogenetic architectures (legacy shim)
from . import filters     ## sample filtering (legacy shim)
from . import mate        ## mate assignment (legacy shim)
from . import reproduce   ## sexual reproduction and phenotypic transmission
from . import founders    ## creation / import of founder haplotypes
from . import ped         ## pedigree objects
from . import sim         ## simulation object (legacy shim)
from . import stats       ## estimation (legacy shim)
from . import proc        ## post-processing (legacy shim)
from . import io          ## input/output
from . import neffect     ## new effect specs
from . import narch       ## new architecture system
from . import parser      ## formula parser
from . import nmate       ## new mate assignment
from . import nfilter     ## new filters (trio, sib-pair)
from . import nstats      ## new statistics
from . import ngwas       ## GWAS and PGS
from . import nsim        ## new simulation loop
from . import legacy      ## legacy modules (arch, sim, mate, etc.)
from . import cli         ## command-line interface

