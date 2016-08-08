from __future__ import print_function

# import os
# import glob

# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd
import numpy as np

# # from scipy.stats import xxx

# from sklearn.preprocessing import scale
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, RandomizedPCA
# from sklearn.cluster import AggromerativeClustering


class Primo(object):
    """Class for single-cell RNA-seq analysis

    explanations

    Attributes
    ----------
    genes : list
        genes to be analyzed
    cells : list
        cells to be analyzed

    Methods
    -------
    load_scRNAseq_data(path_or_dataframe, from_file=True)
        xxxx
    remove_outlier_cells(val)
        xxx
    normalize(normalize_factor=None)
        xxx

    """

    def __init__(self):
        """init documentation"""
        pass

    def load_scRNAseq_data(self, path_or_dataframe, from_file=True):
        """Load single cell RNA-seq data.

        Parameters
        ----------
        path_or_dataframe : str
            DigitalExpression data
        from_file : bool
            If `True` load data from tab-separated text

        Return
        -------
        self : object
            Returns the instance itself
        """

        if from_file:
            self.df_rnaseq = pd.read_csv(path_or_dataframe,
                                         sep="\t", index_col=0)
        else:
            self.df_rnaseq = path_or_dataframe

        self._update_info()

        return self

    def remove_outlier_cells(self, val):

        """Remove outlier cells

        Parameters
        ----------
        val : float
            If number of detected detected transcripts is not in
            range of mean +- `val` * standard deviation,
            the cells are ignored for further analyses.

        Return
        -------
        self : object
            Retunrs the instance itself

        """

        mean = self.df_rnaseq.sum().mean()
        sd = self.df_rnaseq.sum().std()
        index_not_outlier = abs(self.df_rnaseq.sum() - mean) < (val * sd)

        self.df_rnaseq = self.df_rnaseq.ix[:, index_not_outlier]
        self._update_info()

        return self

    def normalize(self, normalize_factor=None):
        """Normalize dataframe

        Parameters
        ----------
        normaize_factor : None, or int
            Total transccipts is normalized to this value.
            If `None`, total of total transcripts is normalized to
            the mean of that for all cells

        Return
        ------
        self : object
            Returns the instance itself

        """

        if normalize_factor is None:
            normalize_factor = self.df_rnaseq.sum().mean()

        self.df_rnaseq = (1.0 * normalize_factor * self.df_rnaseq
                          / self.df_rnaseq.sum())

        self._update_info()

        return self

    def filter_variable_genes(self, z):
        """Filters genes which coefficient of variation are high.

        Parameters
        ----------
        z : float
            z score for filtering

        Return
        ------
        self : object
            Returns the instance itself
        """
        df_log = np.log(self.df_rnaseq + 0.1)

    def _update_info(self):

        self.genes = list(self.df_rnaseq.index)
        self.cells = list(self.df_rnaseq.columns)

        return self


if __name__ == '__main__':
    p = Primo()
    (p.load_scRNAseq_data("./data/test.txt").
     remove_outlier_cells(2.0).
     normalize())
