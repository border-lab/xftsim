import importlib.resources

import pandas as pd


def get_ceu_map():
    """
    Load the CEU haplotype map.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the CEU haplotype map.

    """
    resource = importlib.resources.files(__package__).joinpath('maps/ceu.hg19.map')
    with resource.open('rb') as stream:
        return pd.read_csv(stream)
