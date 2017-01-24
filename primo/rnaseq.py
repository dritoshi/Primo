from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from distutils.version import StrictVersion

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import plot_tsne

__all__ = ['RNAseq', ]


class RNAseq(object):
    """Class for single-cell RNA-seq analysis

    explanations

    Attributes
    ----------
    genes_ : list
        genes to be analyzed
    cells_ : list
        cells to be analyzed
    num_genes_ : int
        number of genes to be analyzed
    num_cells_ : int
        number of cells to be analyzed
    num_genes_variable_ : int
        number of variable genes to be analyzed
    df_rnaseq_ : DataFrame
        Gene expression matrix
    df_rnaseq_variable_ : DataFrame
        Gene expression matrix (only variable genes)
    """

    def __init__(self):
        """init documentation"""
        pass

    def load_scRNAseq_data(self, path_or_dataframe, num_stamp=None,
                           from_file=True, annotation_type="symbol"):
        """Load single cell RNA-seq data.

        Parameters
        ----------
        path_or_dataframe : str
            DigitalExpression data
        num_stamp : int
            Number of STAMP
        from_file : bool
            If `True` load data from tab-separated text
        annotation_type : str
            Type of gene annotation.
            Examples: symbol, uid (Unigene ID)

        Return
        ------
        self : object
            Returns the instance itself

        """
        if from_file:
            self.df_rnaseq_ = pd.read_csv(path_or_dataframe,
                                          sep="\t", index_col=0)
        else:
            self.df_rnaseq_ = path_or_dataframe

        self.annotation_type_ = annotation_type

        if num_stamp is not None:
            if StrictVersion(pd.__version__) >= "0.17":
                ind_stamp = (self.df_rnaseq_.sum().
                             sort_values(ascending=False).index)[0:num_stamp]
            else:
                ind_stamp = (self.df_rnaseq_.sum().
                             order(ascending=False).index)[0:num_stamp]
            self.df_rnaseq_ = self.df_rnaseq_.ix[:, ind_stamp]

        self._remove_all_zero()
        self._update_info()

        return self

    def remove_spike(self, spike_type="ERCC"):
        """ Remove data for spike RNA

        Parameters
        ----------
        spike_type : str
            For example, 'ERCC'

        Return
        ------
        self : object
            Returns the instance itself
        """
        is_spike = [x.startswith(spike_type) for x in self.df_rnaseq_.index]
        is_gene = [not x.startswith(spike_type) for x in self.df_rnaseq_.index]
        self.df_spike_ = self.df_rnaseq_[is_spike]
        self.df_rnaseq_ = self.df_rnaseq_[is_gene]
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
        ind = self.df_rnaseq_.max(axis=1) > max_count
        self.df_rnaseq_ = self.df_rnaseq_.ix[ind, :]
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
        ind = self.df_rnaseq_.mean(axis=1) < cutoff
        self.df_rnaseq_ = self.df_rnaseq_.ix[ind, :]
        self._remove_all_zero()
        self._update_info()

        return self

    def remove_outlier_cells(self, by="sd", how="both", val_sd=None,
                             val_upper_lim=None, val_lower_lim=None):
        """Remove outlier cells

        Parameters
        ----------
        by : str
            This must be 'sd' or 'value' to cut out outlier cells.
        how : str
            This must be 'both', 'upper', or 'lower'.
        val_sd : int
            This value is multiplied by standard deviation of detected
            transcripts.
        val_upper_lim : int
            Upper limit.
        val_lower_lim : int
            Lower limit.

        Return
        ------
        self : object
            Returns the instance itself

        """
        if by == "sd":
            mean = self.df_rnaseq_.sum().mean()
            sd = self.df_rnaseq_.sum().std()
            upper_lim = np.int(mean + val_sd * sd)
            lower_lim = np.int(mean - val_sd * sd)
        elif by == "value":
            if how != "upper":
                lower_lim = val_lower_lim
            if how != "lower":
                upper_lim = val_upper_lim
        else:
            print("Parameter 'by' must be 'sd' or 'value'.")

        if how == "upper":
            index_not_outlier = self.df_rnaseq_.sum() < upper_lim
        elif how == "lower":
            index_not_outlier = self.df_rnaseq_.sum() > lower_lim
        elif how == "both":
            index_not_outlier = ((lower_lim < self.df_rnaseq_.sum()) &
                                 (self.df_rnaseq_.sum() < upper_lim))
        else:
            print("Parameter 'how' must be 'both', 'upper', or 'lower'.")

        self.df_rnaseq_ = self.df_rnaseq_.ix[:, index_not_outlier]
        self._remove_all_zero()
        self._update_info()

        return self

    def remove_cells_no_wish(self, w_obj):
        """Remove cells with which WISH gene expression is all zero

        Parameters
        ----------
        W : :obj: `w_obj`
            primo.wish.Wish instance.

        Return
        ------
        self : object
            Returns the instance itself.
        """
        genes = w_obj.genes_
        index_not_all_zero_wish = (self.df_rnaseq_.ix[genes, :].sum() != 0)

        self.df_rnaseq_ = self.df_rnaseq_.ix[:, index_not_all_zero_wish]
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
            normalize_factor = self.df_rnaseq_.sum().mean()

        self.df_rnaseq_ = (1.0 * normalize_factor *
                           self.df_rnaseq_ / self.df_rnaseq_.sum())

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
        ind = self.df_rnaseq_.max(axis=1) > max_count
        df_log = np.log10(self.df_rnaseq_.ix[ind, :] + 0.1)

        bin_label = ["bin" + str(i) for i in range(1, bin_num + 1)]
        df_log['bin'] = pd.qcut(df_log.mean(axis=1), bin_num, labels=bin_label)

        self.variable_genes_ = []

        if stack is True:
            for binname in bin_label:
                df_log_bin = df_log[df_log.bin == binname].ix[
                    :, :df_log.shape[1]-1]
                df_stack = df_log_bin.stack()
                df_z = ((df_stack - df_stack.mean()) /
                        df_stack.std()).unstack()
                variable_genes_bin = df_log_bin.index[
                    df_z.var(axis=1) > z_cutoff].values
                self.variable_genes_.extend(variable_genes_bin)
        else:
            for binname in bin_label:
                df_log_bin = df_log[df_log.bin == binname].ix[
                    :, :df_log.shape[1]-1]
                dispersion_measure = (df_log_bin.std(axis=1) /
                                      df_log_bin.mean(axis=1)) ** 2
                z_disp = ((dispersion_measure - dispersion_measure.mean()) /
                          dispersion_measure.std())
                variable_genes_bin = df_log_bin.index[z_disp > z_cutoff].values
                self.variable_genes_.extend(variable_genes_bin)

        self.df_rnaseq_variable_ = self.df_rnaseq_.ix[self.variable_genes_, :]
        self.num_genes_variable_ = self.df_rnaseq_variable_.shape[0]

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
            ind = self.df_rnaseq_.index.isin(self.df_rnaseq_variable_.index)

        mean = self.df_rnaseq_.mean(axis=1)
        std = self.df_rnaseq_.std(axis=1)
        cv2 = (std / mean) ** 2
        cv2_poisson = 1. / mean

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(mean, cv2, c="lightgray", marker=".", edgecolors="none",
                   label="all")
        ax.scatter(mean, cv2_poisson, c="green", marker=".", edgecolors="none",
                   label="poisson")
        if colorize_variable_genes:
            ax.scatter(mean[ind], cv2[ind], c="magenta",
                       marker=".", edgecolors="none",
                       label="variable")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** -2, 10 ** 2.5)
        ax.set_ylim(10 ** -2, 10 ** 2.5)
        ax.set_xlabel("Mean")
        ax.set_ylabel("Squared coefficient of variation")

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_file)

        return self

    def pca(self, adding_factor=0.1, **kwargs):
        """PCA

        Parameters
        ----------
        adding_factor : float
            Log transformation will be done after addition of adding factor
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Return the instance itself
        """
        self.df_rnaseq_log_ = np.log(self.df_rnaseq_ + adding_factor)
        self.df_rnaseq_scale_ = pd.DataFrame(
            scale(self.df_rnaseq_log_, axis=1),
            index=self.df_rnaseq_.index,
            columns=self.df_rnaseq_.columns)
        self.df_pca_ = PCA(**kwargs).fit_transform(
            self.df_rnaseq_scale_.ix[self.variable_genes_, :].T)

        return self

    def tsne(self, plot=False, output_dir=None, **kwargs):
        """t-SNE

        Parameters
        ----------
        plot : bool
            if `True`, export PNG file
        output_dir : :obj:`str`, optional
            if plot is `True`, export PNG file to this directory
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Returns the instance itself

        """
        self.tsne_rnaseq_cells_ = TSNE(**kwargs).fit_transform(
            1 - self.df_rnaseq_variable_.corr())
        self.tsne_rnaseq_genes_ = TSNE(**kwargs).fit_transform(
            1 - self.df_rnaseq_variable_.T.corr())

        if plot is True:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes = axes.flatten()
            plot_tsne(self.tsne_rnaseq_cells_, ax=axes[0])
            plot_tsne(self.tsne_rnaseq_genes_, ax=axes[1])
            axes[0].set_title("t-SNE: cells")
            axes[1].set_title("t-SNE: genes")
            plt.tight_layout()
            output_file = os.path.join(output_dir, "tsne_rnaseq.png")
            plt.savefig(output_file)

        return self

    def colorrize_gene(self):
        pass

    def _remove_all_zero(self):
        genes_not_all_zero = (self.df_rnaseq_.sum(axis=1) != 0)
        cells_not_all_zero = (self.df_rnaseq_.sum(axis=0) != 0)
        self.df_rnaseq_ = self.df_rnaseq_.ix[genes_not_all_zero,
                                             cells_not_all_zero]

        return self

    def _update_info(self):
        self.genes_ = list(self.df_rnaseq_.index)
        self.num_genes_ = len(self.genes_)
        self.cells_ = list(self.df_rnaseq_.columns)
        self.num_cells_ = len(self.cells_)

        return self
