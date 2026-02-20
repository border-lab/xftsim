from importlib import resources

import pandas as pd


def get_ceu_map():
    """
    Load the CEU haplotype map.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the CEU haplotype map.

    """
    map_file = resources.files("xftsim") / "maps" / "ceu.hg19.map"
    with resources.as_file(map_file) as path:
        return pd.read_csv(path)
