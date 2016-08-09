from __future__ import print_function

import os
from distutils.version import StrictVersion
# import glob

import pandas as pd
import numpy as np

# # from scipy.stats import xxx

# from sklearn.preprocessing import scale
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, RandomizedPCA
# from sklearn.cluster import AggromerativeClustering

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns


class Primo(object):
    """Class for single-cell RNA-seq analysis

    explanations

    Attributes
    ----------
    genes : list
        genes to be analyzed
    cells : list
        cells to be analyzed
    num_genes : int
        number of genes to be analyzed
    num_cells : int
        number of cells to be analyzed
    num_genes_variable : int
        number of variable genes to be analyzed
    df_rnaseq : DataFrame
        Gene expression matrix
    df_rnaseq_variable : DataFrame
        Gene expression matrix (only variable genes)

    Methods
    -------
    load_scRNAseq_data(path_or_dataframe, from_file=True)
        xxx
    remove_gene_toohigh(cutoff)
        xxx
    remove_gene_maxlessthan(max_count)
        xxx
    remove_outlier_cells(val)
        xxx
    normalize(normalize_factor=None)
        xxx
    filter_variable_genes(z_cutoff, max_count=5)
        xxx
    """

    def __init__(self):
        """init documentation"""
        pass

    def load_scRNAseq_data(self, path_or_dataframe, num_stamp=None,
                           from_file=True):
        """Load single cell RNA-seq data.

        Parameters
        ----------
        path_or_dataframe : str
            DigitalExpression data
        num_stamp : int
            Number of STAMP
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

        if num_stamp is not None:
            if StrictVersion(pd.__version__) >= "0.17":
                ind_stamp = (self.df_rnaseq.sum().
                             sort_values(ascending=False).index)[0:num_stamp]
            else:
                ind_stamp = (self.df_rnaseq.sum().
                             order(ascending=False).index)[0:num_stamp]
            self.df_rnaseq = self.df_rnaseq.ix[:, ind_stamp]

        self._remove_all_zero()
        self._update_info()

        return self

    def remove_gene_maxlessthan(self, max_count):
        """ (Deprecated) Remove genes of which expression max is less than `max_count`.

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
        """Remove genes of which expression mean is highre than `cutoff`.

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

    def filter_variable_genes(self, z_cutoff=2, max_count=5, bin_num=200,
                              stack=False):
        """Filters genes which coefficient of variation are high.

        Parameters
        ----------
        z_cutoff : float
            z score for filtering
        max_count : float
            Genes of which expression max is less than `max_count` are
            filterd out.
        bin_num : int
            number of bin
        stack : bool
            If `True`, expression of genes in the bin are stacked before
            scaling

        Return
        ------
        self : object
            Returns the instance itself

        """
        ind = self.df_rnaseq.max(axis=1) > max_count
        df_log = np.log10(self.df_rnaseq.ix[ind, :] + 0.1)

        bin_label = ["bin" + str(i) for i in range(1, bin_num + 1)]
        df_log['bin'] = pd.qcut(df_log.mean(axis=1), bin_num, labels=bin_label)

        self.variable_genes = []

        if stack is True:
            for binname in bin_label:
                df_log_bin = df_log[df_log.bin == binname].ix[
                    :, :df_log.shape[1]-1]
                df_stack = df_log_bin.stack()
                df_z = ((df_stack - df_stack.mean()) /
                        df_stack.std()).unstack()
                variable_genes_bin = df_log_bin.index[
                    df_z.var(axis=1) > z_cutoff].values
                self.variable_genes.extend(variable_genes_bin)
        else:
            for binname in bin_label:
                df_log_bin = df_log[df_log.bin == binname].ix[
                    :, :df_log.shape[1]-1]
                dispersion_measure = (df_log_bin.std(axis=1) /
                                      df_log_bin.mean(axis=1)) ** 2
                z_disp = ((dispersion_measure - dispersion_measure.mean()) /
                          dispersion_measure.std())
                variable_genes_bin = df_log_bin.index[z_disp > z_cutoff].values
                self.variable_genes.extend(variable_genes_bin)

        self.df_rnaseq_variable = self.df_rnaseq.ix[self.variable_genes, :]
        self.num_genes_variable = self.df_rnaseq_variable.shape[0]

        return self

    def plot_cv(self, output_dir, colorize_variable_genes=True):
        """Plot Mean-CV plot

        Parameters
        ----------
        output_dir : str
            path of output directory
        colorize_variable_genes : bool
            If `True`, colorize variable genes

        Return
        ------
        self : object
            Returns the instance itself

        """
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_file = os.path.join(output_dir, "mean_cv_plot.png")

        if colorize_variable_genes is True:
            ind = self.df_rnaseq.index.isin(self.df_rnaseq_variable.index)

        mean = self.df_rnaseq.mean(axis=1)
        std = self.df_rnaseq.std(axis=1)
        cv2 = (std / mean) ** 2
        cv2_poisson = 1. / mean

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(mean, cv2, c="lightgray", marker=".", edgecolors="none")
        ax.scatter(mean, cv2_poisson, c="green", marker=".", edgecolors="none")
        if colorize_variable_genes:
            ax.scatter(mean[ind], cv2[ind], c="magenta",
                       marker=".", edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** -2, 10 ** 2.5)
        ax.set_ylim(10 ** -2, 10 ** 2.5)
        ax.set_xlabel("Mean")
        ax.set_ylabel("CV ** 2")

        plt.savefig(output_file)

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
    pass

    p = Primo()
    (p.load_scRNAseq_data("../data/St13_1st_dge.txt.gz", num_stamp=1200).
     remove_gene_toohigh(500).
     remove_outlier_cells(2.0).
     normalize())

    p.filter_variable_genes(z_cutoff=1.2, max_count=5,
                            bin_num=2000, stack=True)
    p.plot_cv("../results")
