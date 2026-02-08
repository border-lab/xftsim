import pandas as pd
import pkg_resources


def get_ceu_map():
    """
    Load the CEU haplotype map.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the CEU haplotype map.

    """
    stream = pkg_resources.resource_stream(__name__, 'maps/ceu.hg19.map')

    return pd.read_csv(stream)
