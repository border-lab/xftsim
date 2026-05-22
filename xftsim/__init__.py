import sys
import warnings

import numpy as np

if sys.version_info < (3, 12):
    warnings.warn(
        f"xftsim recommends Python >= 3.12 (you have {sys.version_info.major}.{sys.version_info.minor}). "
        "Some features may not work correctly on older versions.",
        stacklevel=2,
    )

__version__="0.9a.dev110"


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
from . import index       ## indexing
from . import struct      ## data structures
from . import reproduce   ## sexual reproduction and phenotypic transmission
from . import founders    ## creation / import of founder haplotypes
from . import mate        ## mate assignment
from . import ped         ## pedigree objects
from . import io          ## input/output
from . import effect      ## effect specs
from . import arch        ## architecture system
from . import parser      ## formula parser
from . import filters     ## filters (trio, sib-pair)
from . import stats       ## statistics
from . import gwas        ## GWAS and PGS
from . import sim         ## simulation loop
from . import cli         ## command-line interface

