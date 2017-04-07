from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from distutils.version import StrictVersion
from itertools import cycle
from collections import Counter
import multiprocessing as mp

import pandas as pd
import numpy as np
from scipy.stats import entropy, chi2

import seaborn as sns
sns.set_style("white")

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import fdrcorrection0

# from .utils import plot_tsne

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
    spike_type_ : str
        Type of spike RNA. ex) "ERCC"
    df_spike_ : DataFrame
        Gene expression matrix for spike RNA
    df_rnaseq_not_norm_ : DataFrame
        Gene expression matrix, not normalized
    df_rnaseq_log_ : DataFrame
        Gene expression matrix (log2 transformed)
    df_rnaseq_scale_ : DataFrame
        Gene expression matrix (scaled after log2 transformation)
    df_rnaseq_scale_genes_ : DataFrame
        Scaled values on genes
    df_pca_scores_ : DataFrame
        PC scores for cells
    df_pca_components_ : DataFrame
        PC components for cells
    df_pca_scores_genes_ : DataFrame
        PC scores for genes
    df_tsne_rnaseq_cells_ : DataFrame
        t-SNE for cells
    df_tsne_rnaseq_genes_ : DataFrame
        t-SNE for genes
    cluster_label_ : list
        resulted list of clustering
    df_factor_loading_ : DataFrame
        Factor loading in PCA, genes.
    df_facs_ : DataFrame
        FACS channel values
    """

    def __init__(self):
        """init documentation"""
        pass

    def load_scRNAseq_data(self, path_or_dataframe, num_stamp=None,
                           from_file=True, label=None,
                           annotation_type="symbol", spike_type=None):
        """Load single cell RNA-seq data.

        Parameters
        ----------
        path_or_dataframe : str
            DigitalExpression data
        num_stamp : int
            Number of STAMP
        from_file : bool
            If `True` load data from tab-separated text
        label : str
            Experimental label. ex.) plate, replication
        annotation_type : str
            Type of gene annotation.
            Examples: symbol, uid (Unigene ID)
        spike_type : str
            Name of spike used in this experiment. (Ex. "ERCC")

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

        self.label_ = label

        if label is not None:
            self.df_rnaseq_.columns = [x + "_label:" + str(self.label_)
                                       for x
                                       in self.df_rnaseq_.columns]
        if spike_type is None:
            self.spike_type_ = None
        else:
            self.spike_type_ = str(spike_type)
            is_spike = [x.startswith(spike_type)
                        for x in self.df_rnaseq_.index]
            is_gene = [not x.startswith(spike_type)
                       for x in self.df_rnaseq_.index]
            self.df_spike_ = self.df_rnaseq_[is_spike]
            self.df_rnaseq_ = self.df_rnaseq_[is_gene]

        self._remove_all_zero()
        self._update_info()

        return self

    def load_multiple_scRNAseq_data(self, list_path, list_label,
                                    list_num_stamp=None,
                                    annotation_type="symbol", spike_type=None):
        """Load multiple single cell RNA-seq data.
        Data can be imported from tab-separated files.

        Parameters
        ----------
        list_path : list
            list of DigitalExpression file, tab-separated
        list_label : list
            list of sample label, str
        list_num_stamp : list
            list of STAMPs number, if Droplet-type seq.
        annotation_type : str
            Type of gene annotation.
            Examples: symbol, uid (Unigene ID)
        spike_type : str
            Name of spike used in this experiment. (Ex. "ERCC")

        Return
        ------
        self : object
            Returns the instance itself
        """

        if len(list_path) != len(list_label):
            print("Length of list_path and list_label must be the same.")
            return None

        df_rnaseq = pd.DataFrame()

        for i, label in enumerate(list_label):
            tmp_df = pd.read_csv(list_path[i], sep="\t", index_col=0)
            cell_name = [x + "_label:" + label for x in tmp_df.columns]
            tmp_df.columns = cell_name

            if list_num_stamp is not None:
                if StrictVersion(pd.__version__) >= "0.17":
                    ind_stamp = (tmp_df.sum().
                                 sort_values(ascending=False)
                                 .index)[0:list_num_stamp[i]]
                else:
                    ind_stamp = (tmp_df.sum().
                                 order(ascending=False)
                                 .index)[0:list_num_stamp[i]]
                tmp_df = tmp_df.ix[:, ind_stamp]

            df_rnaseq = df_rnaseq.append(tmp_df.T, ignore_index=False)

        self.df_rnaseq_ = df_rnaseq.T.fillna(0)

        self.annotation_type_ = annotation_type

        if spike_type:
            self.spike_type_ = str(spike_type)
            is_spike = [x.startswith(spike_type)
                        for x in self.df_rnaseq_.index]
            is_gene = [not x.startswith(spike_type)
                       for x in self.df_rnaseq_.index]
            self.df_spike_ = self.df_rnaseq_[is_spike]
            self.df_rnaseq_ = self.df_rnaseq_[is_gene]
        else:
            self.spike_type_ = None

        self._remove_all_zero()
        self._update_info()

        return self

    def overview(self):
        """Overview RNAseq data
        """

        if self.spike_type_ is None:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        axes = axes.flatten()

        num_detected_transcript = self.df_rnaseq_.sum().values
        sns.distplot(num_detected_transcript, ax=axes[0])
        axes[0].set_title("Number of detected transcript", fontsize=16)
        axes[0].set_xlim(0,)

        num_detected_gene = (self.df_rnaseq_ > 0).sum().values
        sns.distplot(num_detected_gene, ax=axes[1])
        axes[1].set_title("Number of detected gene", fontsize=16)
        axes[1].set_xlim(0,)

        entr = [entropy(self.df_rnaseq_.iloc[:, i].values)
                for i in range(self.df_rnaseq_.shape[1])]
        if not np.isinf(min(entr)):
            sns.distplot(entr, ax=axes[2])
            axes[2].set_title("Entropy", fontsize=16)

        if self.spike_type_:
            num_detected_transcript_spike = self.df_spike_.sum().values
            sns.distplot(num_detected_transcript_spike, ax=axes[3])
            axes[3].set_title("Number of detected transcript (spike)",
                              fontsize=16)
            axes[3].set_xlim(0,)

            num_detected_gene_spike = (self.df_spike_ > 0).sum().values
            sns.distplot(num_detected_gene_spike, ax=axes[4])
            axes[4].set_title("Number of detected gene (spike)", fontsize=16)
            axes[4].set_xlim(0,)

            entr_spike = [entropy(self.df_spike_.iloc[:, i].values)
                          for i in range(self.df_spike_.shape[1])]
            if not np.isinf(min(entr_spike)):
                sns.distplot(entr_spike, ax=axes[5])
                axes[5].set_title("Entropy (spike)", fontsize=16)

        fig.tight_layout()

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

    def remove_outlier_cells(self, on="entropy", by="sd", how="both",
                             val_sd=None, val_upper_lim=None,
                             val_lower_lim=None):
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
        if on == "entropy":
            target = [entropy(self.df_rnaseq_.iloc[:, i].values)
                      for i in range(self.df_rnaseq_.shape[1])]
            target = np.array(target)
        elif on == "transcript":
            target = self.df_rnaseq_.sum().values
        elif on == "gene":
            target = (self.df_rnaseq_ > 0).sum().values
        else:
            print("Parameter 'on' must be 'entropy', 'transcript' or 'gene'")

        if by == "sd":
            mean = target.mean()
            sd = target.std()
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
            index_not_outlier = target < upper_lim
        elif how == "lower":
            index_not_outlier = target > lower_lim
        elif how == "both":
            index_not_outlier = ((lower_lim < target) &
                                 (target < upper_lim))
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

    def normalize(self, execute=True, normalize_factor=None):
        """Normalize or not normalize dataframe

        Parameters
        ----------
        execute : bool
            If normalization is needed, this must be True.
            If normalization is not needed, this must be False.
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

        self.df_rnaseq_not_norm_ = self.df_rnaseq_

        if execute:
            self.df_rnaseq_ = (1.0 * normalize_factor *
                               self.df_rnaseq_ / self.df_rnaseq_.sum())

        self._remove_all_zero()
        self._update_info()

        return self

    def filter_variable_genes_z(self, z_cutoff=2, max_count=5, bin_num=200,
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

    def filter_variable_genes_line(self, threshold=0.5):
        """Filter variable genes by line.

        Parameters
        ----------
        threshold : float
            Genes are identified as variable genes of:
            log10(CV ** 2) + log10(MEAN) > threshold

        Return
        ------
        self : object
            Returns the instance itself.
        """

        mean = self.df_rnaseq_.mean(axis=1)
        std = self.df_rnaseq_.std(axis=1)
        cv = 1.0 * std / mean
        self.variable_genes_ = list(self.df_rnaseq_.index[
            (np.log10(cv ** 2) + np.log10(mean)) > threshold])

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

    def pca(self, adding_factor=0.1, used_genes='variable', **kwargs):
        """PCA

        Parameters
        ----------
        adding_factor : float
            Log transformation will be done after addition of adding factor
        used_genes : str
            Genes used for PCA. 'variable' or 'all' can be chosen .
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Return the instance itself
        """

        if used_genes is 'variable':
            used_genes = self.variable_genes_
        elif used_genes is 'all':
            used_genes = self.genes_
        else:
            raise ValueError("used_genes must be 'variable' or 'all'")

        self.df_rnaseq_log_ = np.log(self.df_rnaseq_ + adding_factor)
        self.df_rnaseq_scale_ = pd.DataFrame(
            scale(self.df_rnaseq_log_, axis=1),
            index=self.df_rnaseq_.index,
            columns=self.df_rnaseq_.columns)

        # PCA for cells
        X = self.df_rnaseq_scale_.ix[used_genes, :].T
        pca = PCA(**kwargs)
        pca.fit(X)
        scores = pca.transform(X)
        components = pca.components_.T

        pc_name = ["PC" + str(i+1) for i in range(scores.shape[1])]
        self.df_pca_scores_ = pd.DataFrame(scores,
                                           index=self.df_rnaseq_.columns,
                                           columns=pc_name)

        self.df_pca_components_ = pd.DataFrame(components,
                                               index=used_genes,
                                               columns=pc_name)

        # PCA for genes
        self.df_rnaseq_scale_genes_ = pd.DataFrame(
            scale(self.df_rnaseq_log_, axis=0),
            index=self.df_rnaseq_.index,
            columns=self.df_rnaseq_.columns)

        X_genes = self.df_rnaseq_scale_genes_.ix[used_genes, :]
        pca_genes = PCA(**kwargs)
        pca_genes.fit(X_genes)
        scores_genes = pca_genes.transform(X_genes)
        components_genes = pca_genes.components_.T

        pc_name_genes = ["PC" + str(i+1) for i
                         in range(scores_genes.shape[1])]
        self.df_pca_scores_genes_ = pd.DataFrame(scores_genes,
                                                 index=used_genes,
                                                 columns=pc_name_genes)

        self.df_pca_components_genes_ = pd.DataFrame(
            components_genes, index=self.df_rnaseq_.columns, columns=pc_name)

        return self

    def tsne(self, plot=False, output_dir=None, additional=None, **kwargs):
        """t-SNE

        Parameters
        ----------
        plot : bool
            if `True`, export PNG file
        output_dir : :obj:`str`, optional
            if plot is `True`, export PNG file to this directory
        additional : str
            if tSNE for genes are needed, set "gene".
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Returns the instance itself
        """

        X = TSNE(**kwargs).fit_transform(self.df_pca_scores_)
        self.df_tsne_rnaseq_cells_ = pd.DataFrame(
            X, index=self.df_pca_scores_.index, columns=['Dim1', 'Dim2'])

        if additional == "gene":
            X_genes = TSNE(**kwargs).fit_transform(self.df_pca_scores_genes_)
            self.df_tsne_rnaseq_genes_ = pd.DataFrame(
                X_genes, index=self.df_pca_scores_genes_.index,
                columns=['Dim1', 'Dim2'])

        if plot is True:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()

            axes[0].set_xlabel("Dim1", fontsize=14)
            axes[0].set_ylabel("Dim2", fontsize=14)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[0].scatter(self.df_tsne_rnaseq_cells_.iloc[:, 0],
                            self.df_tsne_rnaseq_cells_.iloc[:, 1],
                            c="black",
                            alpha=0.5,
                            s=20,
                            edgecolor='None')

            axes[0].set_title("t-SNE: cells", fontsize=24)

            if additional == "gene":
                axes[1].set_xlabel("Dim1", fontsize=14)
                axes[1].set_ylabel("Dim2", fontsize=14)
                axes[1].set_xticks([])
                axes[1].set_yticks([])
                axes[1].scatter(self.df_tsne_rnaseq_genes_.iloc[:, 0],
                                self.df_tsne_rnaseq_genes_.iloc[:, 1],
                                c="black",
                                alpha=0.5,
                                s=20,
                                edgecolor='None')
                axes[1].set_title("t-SNE: genes", fontsize=24)
            else:
                axes[1].set_xticks([])
                axes[1].set_yticks([])

            plt.tight_layout()
            output_file = os.path.join(output_dir, "tSNE.png")
            plt.savefig(output_file)

        return self

    def colorize_pcs(self, output_dir, num_pc=16):
        """Colorize PC scores in tSNE space

        Parameters
        ----------
        num_pc : int
            Number of PC to be plotted.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        if num_pc > self.df_pca_scores_.shape[1]:
            print("Number of PC was smaller than num_pc.")
            num_pc = self.df_pca_scores_.shape[1]

        ncol = 4
        nrow = np.int(np.ceil(num_pc * 1.0 / ncol))

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 5))
        axes = axes.flatten()

        for i in range(num_pc):
            color = self.df_pca_scores_.ix[:, i].values
            title = "PC" + str(i+1)
            X = self.df_tsne_rnaseq_cells_.iloc[:, 0],
            Y = self.df_tsne_rnaseq_cells_.iloc[:, 1],
            axes[i].set_xlabel("Dim1", fontsize=14)
            axes[i].set_ylabel("Dim2", fontsize=14)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

            axes[i].scatter(X, Y, c=color,
                            cmap=plt.cm.jet,
                            s=20,
                            edgecolor='None')

            axes[i].set_title(title, fontsize=24)

        for i in range(num_pc, len(axes)):
            if i >= ncol:
                fig.delaxes(axes[i])

        plt.tight_layout()

        output_file = os.path.join(output_dir, "tSNE_PCscore.png")
        plt.savefig(output_file)

        return self

    def pairplot_pca(self, output_dir, num_pc=16, coloring_gene=None):
        """Pairplot PC scores in tSNE space

        Parameters
        ----------
        num_pc : int
            Number of PC to be plotted.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        if num_pc > self.df_pca_scores_.shape[1]:
            print("Number of PC was smaller than num_pc.")
            num_pc = self.df_pca_scores_.shape[1]

        ncol = num_pc
        nrow = num_pc

        figw = ncol * 4
        figh = nrow * 4

        fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh))

        if coloring_gene is not None:
            color = self.df_rnaseq_scale_.ix[coloring_gene, :]

        for i in range(num_pc):
            pc_name_Y = "PC" + str(i+1)
            Y = self.df_pca_scores_.ix[:, pc_name_Y]
            for j in range(num_pc):
                pc_name_X = "PC" + str(j+1)
                X = self.df_pca_scores_.ix[:, pc_name_X]
                if i == j:
                    axes[i][j].text(0.5, 0.5, pc_name_X,
                                    horizontalalignment="center",
                                    verticalalignment="center",
                                    fontsize=24)
                    axes[i][j].axis('off')
                else:
                    if coloring_gene is not None:
                        axes[i][j].scatter(X, Y, s=20, c=color,
                                           cmap=plt.cm.jet,
                                           edgecolors='None')
                    else:
                        axes[i][j].scatter(X, Y, s=20, c="black",
                                           edgecolors='None', alpha=0.5)
                    axes[i][j].set_xlabel(pc_name_X, fontsize=14)
                    axes[i][j].set_ylabel(pc_name_Y, fontsize=14)

        plt.tight_layout()

        output_file = os.path.join(output_dir, "Pairplot_PCA.png")
        plt.savefig(output_file)

        return self

    def colorize_genes(self, gene_list, output_file, space="tSNE",
                       channel_list=None, coloring="scale_log",
                       plot_colorbar=False, suptitle=None):
        """Colorize gene expression

        Parameters
        ----------
        gene_list : list
            List of genes
        output_file : str
            Name of output file
        space : str
            "tSNE" or "FACS" (Ch1 vs Ch2)
        channel_list : list
            List of FACS channel. Length must be 2.
        coloring : str
            The way of coloring dot
        plot_colorbar : bool
            Plot or not plot colorbar for normalized gene expression
        suptitle : str
            The title of entire figure

        Return
        ------
        self : object
            Returns the instance itself.
        """

        ncol = 4
        nrow = np.int(np.ceil(len(gene_list) * 1.0 / ncol))

        if plot_colorbar:
            figw = ncol * 5 * 1.07
        else:
            figw = ncol * 5

        if suptitle is not None:
            figh = nrow * 5 + 1.5
        else:
            figh = nrow * 5

        fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh))

        axes = axes.flatten()

        for i, gene in enumerate(gene_list):
            if gene not in self.df_rnaseq_.index:
                axes[i].text(0.5, 0.5, "N.D.",
                             horizontalalignment="center",
                             verticalalignment="center",
                             fontsize=24)
                axes[i].set_xlabel("", fontsize=14)
                axes[i].set_ylabel("", fontsize=14)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            else:
                if coloring == "raw_count":
                    color = self.df_rnaseq_not_norm_.ix[gene, :]
                elif coloring == "normalized_count":
                    color = self.df_rnaseq_.ix[gene, :],
                elif coloring == "scale_log":
                    color = self.df_rnaseq_scale_.ix[gene, :],
                else:
                    print("Parameter 'coloring' must be "
                          "scale_log, raw_count or normalized_count.")

                if space == "tSNE":
                    X = self.df_tsne_rnaseq_cells_.iloc[:, 0]
                    Y = self.df_tsne_rnaseq_cells_.iloc[:, 1]
                    axes[i].set_xlabel("Dim1", fontsize=14)
                    axes[i].set_ylabel("Dim2", fontsize=14)
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                elif space == "FACS":
                    channel_x = channel_list[0]
                    channel_y = channel_list[1]
                    X = self.df_facs_.ix[:, channel_x]
                    Y = self.df_facs_.ix[:, channel_y]
                    axes[i].set_xlabel(str(channel_x), fontsize=14)
                    axes[i].set_ylabel(str(channel_y), fontsize=14)
                else:
                    print("Parameter 'space' must be "
                          "tSNE or FACS.")

                S = axes[i].scatter(X, Y, c=color,
                                    cmap=plt.cm.jet,
                                    s=20,
                                    edgecolor='None')

                if plot_colorbar:
                    driver = make_axes_locatable(axes[i])
                    ax_cb = driver.new_horizontal(size="3%", pad=0.05)
                    fig.add_axes(ax_cb)
                    plt.colorbar(S, cax=ax_cb)

            axes[i].set_title(gene, fontsize=24)

        for i in range(len(gene_list), len(axes)):
            if i >= ncol:
                fig.delaxes(axes[i])

        plt.tight_layout()

        if suptitle is not None:
            fig.suptitle(suptitle, x=0, horizontalalignment='left',
                         fontsize=32, fontweight='bold')
            fig.subplots_adjust(top=1-1.5/figh)

        plt.savefig(output_file)

        return self

    def calc_factor_loading(self, output_dir):
        """Calculate factor loading

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance itself
        """

        X = self.df_pca_scores_
        Y = self.df_rnaseq_scale_.T
        Z = corr_inter(X, Y)
        row_name = ["PC" + str(i+1) for i in range(X.shape[1])]
        col_name = Y.columns.values
        self.df_factor_loading_ = pd.DataFrame(Z, index=row_name,
                                               columns=col_name)

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_file = os.path.join(output_dir, "factor_loading.tsv")
        self.df_factor_loading_.T.to_csv(output_file, sep="\t",
                                         index=True, index_label="GeneSymbol")

        return self

    def colorize_correlated_genes(self, num_pc=5, num_gene=8, output_dir=None,
                                  space="tSNE", channel_list=None,
                                  coloring="scale_log", plot_colorbar=False):
        """Colorize gene expression of which are highly correlated
        with PC scores in t-SNE space

        Parameters
        ----------
        num_pc : int
            Number of PC
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Returns the instance itself
        """

        if output_dir is None:
            output_dir = "./"

        if num_pc > self.df_factor_loading_.shape[0]:
            num_pc = self.df_factor_loading_.shape[0]

        for i in range(num_pc):
            pc_name = "PC" + str(i+1)
            fl_sorted = self.df_factor_loading_.ix[pc_name, :].sort_values()

            gene_list = fl_sorted[-1 * num_gene:].index.values
            gene_list = gene_list[::-1]
            val_lim = np.round(fl_sorted[-1 * num_gene], 2)
            suptitle = (pc_name + "-correlated genes (positively, > " +
                        str(val_lim) + ")")
            output_file = os.path.join(output_dir,
                                       space + "_" + pc_name +
                                       "-correlated_positively.png")
            self.colorize_genes(gene_list, output_file, space, channel_list,
                                coloring, plot_colorbar, suptitle)

            gene_list = fl_sorted[:num_gene].index.values
            val_lim = np.round(fl_sorted[num_gene], 2)
            suptitle = (pc_name + "-correlated genes (negatively, < " +
                        str(val_lim) + ")")
            output_file = os.path.join(output_dir,
                                       space + "_" + pc_name +
                                       "-correlated_negatively.png")
            self.colorize_genes(gene_list, output_file, space, channel_list,
                                coloring, plot_colorbar, suptitle)

        return self

    def merge_facs_data(self, facs_instance_list):
        """Merge FACS data to RNAseqdata

        Parameters
        ----------
        facs_instance_list : list
            list of primo.facs instance

        Return
        ------
        self : object
            Returns the instance itself
        """

        df = pd.DataFrame()

        for f in facs_instance_list:
            df = df.append(f.df_barcode_facs_, ignore_index=True)

        df = df[df.barcode.isin(self.cells_)]
        df = df.set_index("barcode")
        df = df.ix[self.cells_, :]
        self.df_facs_ = df

        return self

    def colorize_channel(self, channel_list, output_dir,
                         coloring="raw", plot_colorbar=True):
        """Colorize strength of FACS channel

        Parameters
        ----------
        channel_list : list
            List of channels to be shown
        coloring : str
            The way of coloring. "raw" or "log".

        plot_colorbar : bool
            Plot or not plot colorbar for normalized gene expression

        Return
        ------
        self : object
            Returns the instance itself
        """

        ncol = 3
        nrow = np.int(np.ceil(len(channel_list) * 1.0 / ncol))

        if plot_colorbar:
            figw = ncol * 5 * 1.07
        else:
            figw = ncol * 5

        figh = nrow * 5

        fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh))

        axes = axes.flatten()

        for i, channel in enumerate(channel_list):

            X = self.df_tsne_rnaseq_cells_.iloc[:, 0]
            Y = self.df_tsne_rnaseq_cells_.iloc[:, 1]
            axes[i].set_xlabel("Dim1", fontsize=14)
            axes[i].set_ylabel("Dim2", fontsize=14)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

            if channel not in self.df_facs_.columns:
                axes[i].text(0.5, 0.5, "No channel",
                             horizontalalignment="center",
                             verticalalignment="center",
                             fontsize=24)
            else:
                if coloring == "raw":
                    color = self.df_facs_.ix[:, channel]
                elif coloring == "log":
                    color = np.log(self.df_facs_.ix[:, channel]+0.01),
                else:
                    print("Parameter 'coloring' must be 'raw' or 'log'.")

                S = axes[i].scatter(X, Y, c=color,
                                    cmap=plt.cm.jet,
                                    s=20,
                                    edgecolor='None')

                if plot_colorbar:
                    driver = make_axes_locatable(axes[i])
                    ax_cb = driver.new_horizontal(size="3%", pad=0.05)
                    fig.add_axes(ax_cb)
                    plt.colorbar(S, cax=ax_cb)

            axes[i].set_title(str(channel), fontsize=24)

        for i in range(len(channel_list), len(axes)):
            if i >= ncol:
                fig.delaxes(axes[i])

        plt.tight_layout()
        output_file = os.path.join(output_dir, "tSNE_FACSchannel.png")
        plt.savefig(output_file)

        return self

    def clustering(self, **kwargs):
        """Clustering cells using DBSCAN

        Parameters
        ----------
        **kwargs
            Arbitary keyword arguments.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self.df_tsne_rnaseq_cells_

        dbscan = DBSCAN(**kwargs)

        self.cluster_label_ = dbscan.fit_predict(X)
        self.cluster_label_ += 1

        self.cells_in_cluster_ = dict()

        for lbl in sorted(list(set(self.cluster_label_))):
            cell = (self.cluster_label_ == lbl)
            self.cells_in_cluster_[lbl] = list(
                self.df_tsne_rnaseq_cells_.index[cell])

        return self

    def colorize_label(self, output_dir, label="sample", list_color=None):
        """Colorize dots for each label on t-SNE space

        Parameters
        ----------
        output_dir : str
            Output directory
        label : str
            Label used for coloring. "sample" or "cluster" can be selected.
        list_color : list
            List of manually selected colors for each label

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if label == "sample":
            self.used_color_sample_ = dict()
            used_color = self.used_color_sample_
            list_label = [x.split("_label:")[1] for x
                          in self.df_rnaseq_.columns]
        elif label == "cluster":
            self.used_color_cluster_ = dict()
            used_color = self.used_color_cluster_
            list_label = self.cluster_label_
        else:
            raise ValueError("Option 'label' must be 'sample' or 'cluster'.")

        output_file = os.path.join(output_dir, "tSNE_" + label + ".png")

        series_label = pd.Series(list_label)
        factor_label = list(set(list_label))
        factor_label.sort()

        counter = Counter(list_label)
        counter = pd.Series(counter)
        counter = counter.to_dict()

#         if label == "cluster":
#             list_label = ["cluster" + str(i) for i in list_label]
#             series_label = pd.Series(list_label)
#             factor_label = ["cluster" + str(i) for i in factor_label]

        if list_color is None:
            palette = cycle(sns.color_palette("hls", len(factor_label)))
        else:
            if len(list_color) == len(factor_label):
                palette = cycle(list_color)
            else:
                raise ValueError("Length of list_color must be the same "
                                 "number of labels.")

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.set_xlabel("Dim1", fontsize=14)
        ax.set_ylabel("Dim2", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        for i, lbl in enumerate(factor_label):

            cell = (series_label == lbl)
            X = self.df_tsne_rnaseq_cells_.iloc[cell.values, 0]
            Y = self.df_tsne_rnaseq_cells_.iloc[cell.values, 1]

            if (label == "cluster") & (lbl == 0):
                c = "lightgray"
            else:
                c = next(palette)

            used_color[lbl] = c

            legend_name = str(lbl) + " (" + str(counter[lbl]) + ")"
            ax.scatter(X, Y, c=c, s=5,
                       edgecolors='None', label=legend_name)

        plt.legend(markerscale=3.0, bbox_to_anchor=(1.05, 1),
                   loc=2, borderaxespad=0., title=label)

        plt.savefig(output_file)

        if label == "sample":
            self.used_color_sample_ = used_color
        elif label == "cluster":
            self.used_color_cluster_ = used_color

        return self

    def violinplot(self, output_dir, gene_list=[], count="normalized",
                   label="sample", remove_label=[]):
        """Violinplots for genes

        Parameters
        ----------
        output_dir : str
            Output directory
        gene_list : list
            List of genes
        label : str
            Label used for coloring. "sample" or "cluster" can be selected.
        remove_label : list
            list of labels to be removed for plots.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if count == "normalized":
            df = self.df_rnaseq_.T
        elif count == "raw":
            df = self.df_rnaseq_not_norm_.T
        else:
            raise ValueError("Option 'count' must be 'normalized' or 'raw'.")

        if label == "sample":
            label_name = "Sample name"
            list_label = [x.split("_label:")[1] for x
                          in df.index]
            color_dict = self.used_color_sample_
        elif label == "cluster":
            label_name = "Cluster name"
            list_label = self.cluster_label_
            color_dict = self.used_color_cluster_
        else:
            raise ValueError("Option 'label' must be 'sample' or 'cluster'.")

        df[label] = list_label
        output_file = os.path.join(output_dir, "violinplot_" + label + ".png")

        factor_label = list(set(list_label) - set(remove_label))

        factor_label.sort()

        df = df[df[label].isin(factor_label)]

        palette = [color_dict[l] for l in factor_label]

        if len(gene_list) == 0:
            detected_transcripts = (self.df_rnaseq_not_norm_.
                                    loc[:, df.index].sum())
            detected_genes = (self.df_rnaseq_not_norm_.
                              loc[:, df.index] > 0).sum()
            df['transcripts'] = detected_transcripts
            df['genes'] = detected_genes

            gene_mito = [x for x in self.genes_ if x.startswith("mt-")]
            mito_transcripts = (self.df_rnaseq_not_norm_.
                                ix[gene_mito, df.index].sum())
            mito_ratio = mito_transcripts / detected_transcripts
            df['mt-genes transcripts'] = mito_transcripts
            df['mt-genes ratio'] = mito_ratio

            ncol = 4
            nrow = 1
            figh = nrow * 4
            figw = ncol * 4 + len(factor_label) * 0.5

            fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh))
            axes = axes.flatten()

            for i, column in enumerate(['transcripts', 'genes',
                                        'mt-genes transcripts',
                                        'mt-genes ratio']):
                sns.violinplot(x=label, y=column, data=df, scale="width",
                               linewidth=1, palette=palette, ax=axes[i])
                axes[i].set_ylim(0,)
                axes[i].set_title(column, fontsize="18")
                axes[i].set_xlabel(label_name)

                if i == 3:
                    axes[i].set_ylabel("Ratio")
                else:
                    axes[i].set_ylabel("Number")

        else:
            ncol = 4
            nrow = np.int(np.ceil(len(gene_list) * 1.0 / ncol))
            figh = nrow * 4
            figw = ncol * 4 + len(factor_label) * 0.5

            fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh))
            axes = axes.flatten()

            for i, gene in enumerate(gene_list):
                sns.violinplot(x=label, y=gene, data=df, scale="width",
                               linewidth=1, palette=palette, ax=axes[i])
                axes[i].set_ylim(0,)
                axes[i].set_title(gene, fontsize="24")
                axes[i].set_xlabel(label_name)
                axes[i].set_ylabel("Count")

            for i in range(len(gene_list), len(axes)):
                if i >= ncol:
                    fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(output_file)

        return self

    def calc_markers(self, psuedo_count=1, mean_diff=2.0, fdr=0.05,
                     num_core=16, random_state=12345):
        """Calculate marker genes for clusters

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance itself.
        """

        factor_label = list(set(self.cluster_label_))
        factor_label.sort()

        dict_deg = dict()
        marker_genes = []

        for i, cluster_i in enumerate(factor_label):
            if cluster_i == 0:
                continue
            for j, cluster_j in enumerate(factor_label):
                if cluster_j == 0:
                    continue
                if i < j:
                    res_df = self._compare_two_clusters(
                        cluster_i, cluster_j, psuedo_count, mean_diff,
                        fdr, num_core, random_state)
                    dict_deg[(cluster_i, cluster_j)] = res_df
                    marker_genes.extend(list(res_df.index))

        marker_genes = list(set(marker_genes))

        self.dict_deg = dict_deg

        self.markers = marker_genes

        return self

    def _compare_two_clusters(self, cluster_1, cluster_2, psuedo_count,
                              mean_diff, fdr, num_core, random_state):

        series_label = pd.Series(self.cluster_label_)
        bool_cell_1 = list(series_label == cluster_1)
        bool_cell_2 = list(series_label == cluster_2)

        df = self.df_rnaseq_not_norm_
        df_1 = df.iloc[:, bool_cell_1]
        df_2 = df.iloc[:, bool_cell_2]
        use_gene_1 = df.index[
            (df_1.mean(axis=1) / (psuedo_count + df_2.mean(axis=1)) >
             mean_diff)]
        use_gene_2 = df.index[
            (df_2.mean(axis=1) / (psuedo_count + df_1.mean(axis=1)) >
             mean_diff)]
        use_gene = list(use_gene_1) + list(use_gene_2)

        if len(use_gene) == 0:
            res_df = pd.DataFrame()
            return res_df

        def select_100cells(df, random_state):
            if df.shape[1] > 100:
                random_state = np.random.RandomState(random_state)
                cell = random_state.choice(range(df.shape[1]),
                                           size=100, replace=False)
                df = df.ix[:, cell]
            return df

        df_1 = select_100cells(df_1, random_state)
        df_2 = select_100cells(df_2, random_state)

        def worker(genes, out_q):
            df_delta = pd.DataFrame()

            for gene in genes:
                x = np.round(self._binom_one_gene(df_1, df_2, gene), 2)
                df_delta.loc[gene, 'delta'] = x
            out_q.put(df_delta)

        chunk = np.int(np.ceil(len(use_gene) / num_core))
        genes_sep = [use_gene[i:i+chunk] for i
                     in range(0, len(use_gene), chunk)]

        out_q = mp.Queue()
        jobs = []

        for genes in genes_sep:
            p = mp.Process(target=worker, args=(genes, out_q))
            jobs.append(p)
            p.start()

        res_df = pd.DataFrame()
        for i in range(len(genes_sep)):
            res_df = res_df.append(out_q.get())

        [job.join() for job in jobs]

        res_df = res_df.loc[use_gene, :]
        mean_df_1 = self.df_rnaseq_.ix[use_gene, bool_cell_1].mean(axis=1)
        mean_df_2 = self.df_rnaseq_.ix[use_gene, bool_cell_2].mean(axis=1)
        res_df['mean_cluster' + str(cluster_1)] = mean_df_1
        res_df['mean_cluster' + str(cluster_2)] = mean_df_2
        res_df['FC'] = mean_df_1 / mean_df_2

        p_val = 1 - chi2.cdf(res_df['delta'], df=1)
        q_val = fdrcorrection0(p_val)[1]
        res_df['p_value'] = np.round(p_val, 4)
        res_df['q_value'] = np.round(q_val, 4)

        res_df = res_df[res_df.q_value < fdr]
        res_df = res_df.sort_values(by="FC", ascending=False)

        return res_df

    def _binom_one_gene(self, df_1, df_2, gene):
        df_1 = formatting(df_1, "A", gene)
        df_2 = formatting(df_2, "B", gene)
        df = pd.concat([df_1, df_2], axis=0)
        df.index = range(len(df))
        df['count_not'] = df['count_total'] - df['count_gene']

        dev_null = calc_deviance(
            formula='count_gene + count_not ~ 1', df=df)
        dev = calc_deviance(
            formula='count_gene + count_not ~ 1 + C(condition)', df=df)
        delta = dev_null - dev

        return delta

    def export_dataframes(self, output_dir):
        """Export dataframes

        Parameters
        ----------
        output_dir : str
            Directory where files are exported

        Return
        ------
        self : object
            Returns the instance itself
        """

        if hasattr(self, 'df_rnaseq_not_norm_'):
            output_file = os.path.join(output_dir, "df_count_raw.tsv")
            self.df_rnaseq_not_norm_.astype(int).to_csv(output_file,
                                                        sep="\t", index=True)

        if hasattr(self, 'df_rnaseq_log_'):
            output_file = os.path.join(output_dir, "df_count_log_norm.tsv")
            self.df_rnaseq_log_.to_csv(output_file, sep="\t", index=True)

        if hasattr(self, 'df_rnaseq_scale_'):
            output_file = os.path.join(output_dir, "df_count_scale.tsv")
            self.df_rnaseq_scale_.to_csv(output_file, sep="\t", index=True)

        if hasattr(self, 'df_pca_scores_'):
            output_file = os.path.join(output_dir, "df_pca_scores.tsv")
            self.df_pca_scores_.to_csv(output_file, sep="\t", index=True)

        if hasattr(self, 'df_pca_components_'):
            output_file = os.path.join(output_dir, "df_pca_components.tsv")
            self.df_pca_components_.to_csv(output_file, sep="\t", index=True)

        if hasattr(self, 'df_factor_loading_'):
            output_file = os.path.join(output_dir, "df_pca_factor_loading.tsv")
            self.df_factor_loading_.to_csv(output_file, sep="\t", index=True)

        if hasattr(self, 'df_tsne_rnaseq_cells_'):
            output_file = os.path.join(output_dir, "df_tsne.tsv")
            self.df_tsne_rnaseq_cells_.to_csv(output_file,
                                              sep="\t", index=True)

        if hasattr(self, 'df_facs_'):
            output_file = os.path.join(output_dir, "df_facs.tsv")
            self.df_facs_.to_csv(output_file, sep="\t", index=True)

        return self

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


def corr_inter(X, Y):
    X_normed = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    Y_normed = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=0)
    return np.dot(X_normed.T, Y_normed) / X.shape[0]


def formatting(df, condition, gene):
    df_new = pd.DataFrame()
    df_new['count_gene'] = df.loc[gene]
    df_new['condition'] = str(condition)
    df_new['count_total'] = df.sum()
    return df_new


def calc_deviance(formula, df):
    model = smf.glm(formula, df, family=sm.families.Binomial())
    result = model.fit()
    return result.deviance

# def concatenate_rnaseq_instance(list_rnaseq, list_label):
#    """Concatenate primo.rnaseq instance
#
#    Parameters
#    ----------
#    list_rnaseq : list
#        List of RNAseq instance
#    list_label : list
#        List of experimental label. (ex) plate, replication.
#
#    Return
#    ------
#    r : primo.rnaseq
#        primo.rnaseq instance
#    """
#
#    if len(list_rnaseq) != len(list_label):
#        print("Length of list_rnaseq and list_label must be the same")
#        return None
#
#    df_rnaseq = pd.DataFrame()
#    df_rnaseq_not_norm = pd.DataFrame()
#    df_facs = pd.DataFrame()
#
#    for i, r in enumerate(list_rnaseq):
#        df_rnaseq = df_rnaseq.append(r.df_rnaseq_, ignore_index=True)
#        df_rnaseq_not_norm = df_rnaseq_not_norm.append(
#            r.df_rnaseq_not_norm_, ignore_index=True)
#        df_facs = df_facs.append(r.df_facs_, ignore_index=True)
#
#    pass
