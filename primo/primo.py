from __future__ import print_function

# import os
# import glob

# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd
import numpy as np

# # from scipy.stats import xxx

from sklearn.preprocessing import scale
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

        self._remove_all_zero()
        self._update_info()

        return self

    def remove_gene_maxlessthan(self, max_count):
        """
        (Deprecated)
        Remove genes of which expression max is less than `max_count`.

        Parameters
        ----------
        max_count : float
            Genes of which expression max is less than `max_count` are
            filterd out.

        Return
        ------
        self : object
            Returns the instance itself

        """

        ind = self.df_rnaseq.max(axis=1) > max_count
        self.df_rnaseq = self.df_rnaseq.ix[ind, :]
        self._remove_all_zero()
        self._update_info()

        return self

    def remove_gene_toohigh(self, cutoff):
        """
        Remove genes of which expression mean is highre than `cutoff`.

        Parameters
        ----------
        cutoff : float
            Genes of which expression mean is higher than `cutoff`
            are filtered out.

        Return
        ------
        self : object
            Returns the instance itself

        """
        ind = self.df_rnaseq.mean(axis=1) < cutoff
        self.df_rnaseq = self.df_rnaseq.ix[ind, :]
        self._remove_all_zero()
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
        self._remove_all_zero()
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

        self.df_rnaseq = (1.0 * normalize_factor *
                          self.df_rnaseq / self.df_rnaseq.sum())

        self._remove_all_zero()
        self._update_info()

        return self

    def filter_variable_genes(self, z_cutoff, max_count=5):
        """Filters genes which coefficient of variation are high.

        Parameters
        ----------
        z_cutoff : float
            z score for filtering

        max_count : float
            Genes of which expression max is less than `max_count` are
            filterd out.

        Return
        ------
        self : object
            Returns the instance itself
        """

        ind = self.df_rnaseq.max(axis=1) > max_count
        df_log = np.log(self.df_rnaseq.ix[ind, :] + 0.1)

        bin_num = 2000
        bin_label = ["bin" + str(i) for i in range(1, bin_num + 1)]
        df_log['bin'] = pd.qcut(df_log.mean(axis=1), bin_num, labels=bin_label)

        variable_genes = []
        for binname in bin_label:
            df_log_bin = df_log[df_log.bin == binname].ix[
                :, :df_log.shape[1]-1]
            dispersion_measure = (df_log_bin.var(axis=1) /
                                  df_log_bin.mean(axis=1))
            variable_genes_bin = df_log_bin.index[
                scale(dispersion_measure) > z_cutoff].values
            variable_genes.extend(variable_genes_bin)

        self.df_rnaseq_variable = self.df_rnaseq.ix[variable_genes, :]
        self.num_genes_variable = self.df_rnaseq_variable.shape[0]

        return self

    def _remove_all_zero(self):
        genes_not_all_zero = (self.df_rnaseq.sum(axis=1) != 0)
        cells_not_all_zero = (self.df_rnaseq.sum(axis=0) != 0)
        self.df_rnaseq = self.df_rnaseq.ix[genes_not_all_zero,
                                           cells_not_all_zero]

        return self

    def _update_info(self):
        self.genes = list(self.df_rnaseq.index)
        self.num_genes = len(self.genes)
        self.cells = list(self.df_rnaseq.columns)
        self.num_cells = len(self.cells)

        return self


if __name__ == '__main__':

    p = Primo()
    (p.load_scRNAseq_data("../data/St13_1st_dge.txt.gz").
     remove_gene_toohigh(500).
     remove_outlier_cells(2.0).
     normalize())

    p.filter_variable_genes(z_cutoff=1.7, max_count=5)
