from __future__ import print_function

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# import pandas as pd
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_samples

from primo.utils import plot_tsne

__all__ = ['SpatialExpression', '_make_custum_cmap']


class SpatialExpression(object):
    """Class for spatial gene expression pattern

    Attributes
    ----------
    r_obj_ : :obj:`RNAseq`
        Instance of `RNAseq` class instance from primo.rnaseq.
    w_obj_ : :obj:`Wish`
        Instance of `Wish` class instance from primo.position.
    p_obj_ : :obj:`Position`
        Instance of `Position` class instance from primo.position.
    genes_ : list
        List of all genes. (from r_obj_)
    pixel_names_ : list
        List of pixel names. Example 'pix' + int number
    pixel_ : int
        The length of image (from p_obj_)
    spatial_ : ndarray, shape(len(genes_), pixel)
        Spatial gene expression pattern matrix
    spatial_images_ : :obj:`list` of :obj:`ndarray`
        List of images for spatial gene expression pattern

    """

    def __init__(self):
        pass

    def load_inputs(self, r_obj, w_obj, p_obj):
        """Load input objects

        Parameters
        ----------
        r_obj : :obj:`RNAseq`.
            primo RNAseq class instance.
        w_obj : :obj:`Wish`.
            primo Wish class instance.
        p_obj : :obj:`Position`.
            primo Position class instance.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        self.r_obj_ = r_obj
        self.w_obj_ = w_obj
        self.p_obj_ = p_obj

        return self

    def predict(self):
        """Predict spatial gene expression pattern

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance itself.

        """

        X = self.r_obj_.df_rnaseq_
        Y = self.p_obj_.position_

        self.genes_ = list(X.index)
        self.pixel_names_ = list(Y.columns)

        self.pixel_ = np.int(np.sqrt(Y.shape[1]))

        self.spatial_ = np.dot(X, Y) / np.array(self.p_obj_.mean_position_)
        self.spatial_ = np.nan_to_num(self.spatial_)
        self.spatial_ = (self.spatial_.T / self.spatial_.sum(axis=1)).T

        self.spatial_images_ = [self.spatial_[i].reshape(
            self.pixel_, self.pixel_)
                                for i in range(self.spatial_.shape[0])]

        self.spatial_ = pd.DataFrame(self.spatial_,
                                     index=self.genes_,
                                     columns=self.pixel_names_)

        return self

    def plot_spatial_variable(self, output_dir, is_uid=False,
                              conversion_table_file=None):
        """Plot spatial gene expression pattern

        Parameters
        ----------
        output_dir : str
            Output directory to which png files are exported
        is_uid : bool
            If `True`, gene_list is interpretted as Unegene ID.
        conversion_table_file : str
            If is_uid==`True`, input file path for conversion_table.

        Return
        ------
        self : object
            Returns the instance itself

        """
        # plot for variable genes
        gene_list = self.r_obj_.variable_genes_
        output_file = os.path.join(output_dir, "spatial_variable.png")
        self._plot_image(output_file, gene_list, is_uid,
                         conversion_table_file)

        return self

    def plot_spatial_interest(self, output_dir, gene_list,
                              is_uid=False,
                              conversion_table_file=None):
        """Plot spatial gene expression pattern

        Parameters
        ----------
        output_dir : str
            Output directory to which png files are exported.
        gene_list : :obj:`list` of :obj:`str`
            list of gene symbol or uid (Unigene ID).
        is_uid : bool
            If `True`, gene_list is interpretted as Unegene ID.
        conversion_table_file : str
            If is_uid==`True`, input file path for conversion_table.

        Return
        ------
        self : object
            Returns the instance itself

        """

        output_file = os.path.join(output_dir, "spatial_interest.png")
        self._plot_image(output_file, gene_list, is_uid,
                         conversion_table_file)

        return self

    def predict_loocv(self):
        """Predicts spatial expression pattern for genes using LOOCV

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance it self

        """

        genes = self.p_obj_.genes_
        self.spatial_loocv_ = []
        self.spatial_loocv_images_ = []

        for i, gene in enumerate(genes):
            X = self.r_obj_.df_rnaseq_.ix[gene, :]
            Y = self.p_obj_.position_loocv_[i]
            Y_mean = self.p_obj_.position_loocv_mean_[i]

            spatial = np.dot(X, Y) / np.array(Y_mean)
            spatial = np.nan_to_num(spatial)
            spatial = spatial / spatial.sum()
            spatial_image = spatial.reshape(self.pixel_, self.pixel_)

            self.spatial_loocv_.append(spatial)
            self.spatial_loocv_images_.append(spatial_image)

        return self

    def plot_loocv(self, output_dir):
        """Plots the results of LOOCV

        Parameters
        ----------
        output_dir : str
            The png file is saved in this directory.

        Return
        ------
        self : object
            Returns the instance itself

        """
        output_file = os.path.join(output_dir, "images_loocv.png")

        genes = self.p_obj_.genes_

        ncol = 6
        nrow = np.int(np.ceil(3.0 * len(genes) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten()

        for i, gene in enumerate(genes):
            im1 = self.w_obj_.wish_images_filtered_[
                self.w_obj_.genes_.index(gene)]
            im2 = self.spatial_images_[
                self.genes_.index(gene)]
            im3 = self.spatial_loocv_images_[i]

            axes[i * 3 + 0].imshow(im1, cmap=plt.cm.Purples)
            axes[i * 3 + 0].set_title(gene + " : original", fontsize=10)

            axes[i * 3 + 1].imshow(im2, cmap=plt.cm.jet)
            axes[i * 3 + 1].set_title(gene + " : inference", fontsize=10)

            axes[i * 3 + 2].imshow(im3, cmap=plt.cm.jet)
            axes[i * 3 + 2].set_title(gene + " : loocv", fontsize=10)

        for ax in axes:
            ax.axis('off')

        for ax in axes.ravel():
            if not (len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(output_file)

        return self

    def tsne(self, output_dir, **kwargs):
        """t-SNE

        Calculates t-SNE for two directions and plots the results.

        Parameters
        ----------
        output_dir : str
            If plot is `True`, export PNG file to this directory.
        **kwargs
            Arbitary keyword arguments.

        Return
        ------
        self : object
            Returns the instance itself

        """

        self._calc_tsne(**kwargs)
        self._plot_tsne(output_dir)

        return self

    def clustering(self, n_clusters, algorithm,
                   output_dir, cmap=None, **kwargs):
        """Clustering pixels

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        algorithm : str
            Name of clustering algorithm to be used.
        output_dir : str
            The resulted PNG file is exported to this directory.
        cmap : :obj:``
            Color map.
        **kwargs : kwargs
            Keyword arguments for clustering algorithm.

        Return
        ------
        self : object
            Returns the instance itself.
        """

        if cmap is None:
            cmap = plt.cm.jet

        self.calc_clustering(n_clusters, algorithm, **kwargs)
        self.plot_clustering(output_dir, cmap=cmap)

        return self

    def calc_clustering(self, n_clusters, algorithm, **kwargs):
        """Calculats clustering

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        algorithm : str
            Name of clustering algorithm to be used.
        **kwargs : kwargs
            Keyword arguments for clustering algorithm

        Return
        ------
        self : object
            Returns the instance itself.
        """

        if algorithm == "KMeans":
            method = KMeans(n_clusters=n_clusters, **kwargs)

        elif algorithm == "AgglomerativeClustering":
            connectivity = kneighbors_graph(self.tsne_spatial_pixels_,
                                            n_neighbors=20,
                                            include_self=False)
            method = AgglomerativeClustering(n_clusters=n_clusters,
                                             connectivity=connectivity,
                                             linkage="ward",
                                             **kwargs)

        else:
            print("algorithm parameter should be `KMeans` or "
                  "`AgglomerativeClustering`.")
            raise ValueError

        self.clusters_ = method.fit_predict(self.tsne_spatial_pixels_)
        self.clusters_ = self.clusters_ + 1

        self.df_cluster_ = pd.DataFrame(index=('cluster',),
                                        columns=self.w_obj_.pixel_name_all_)
        self.df_cluster_ = self.df_cluster_.fillna(0)
        self.df_cluster_.ix[
            'cluster', self.w_obj_.pixel_name_embryo_] = self.clusters_

        return self

    def plot_clustering(self, output_dir, cmap):
        """Plot the results of clustering.

        Parameters
        ----------
        output_dir : str
            The png file is exported to this directory.
        cmap : :obj:`LinearSegmentedColormap`
            Color map.

        Return
        ------
        self : object
            Returns the instance itself.
        """

        output_file = os.path.join(output_dir,
                                   "Plot_clustering.png")
        cmap = _make_custum_cmap(cmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()

        self._plot_clusters_image(ax=axes[0], cmap=cmap, mirror=True)
        self._plot_clusters_tsne(ax=axes[1], cmap=cmap)
        self._plot_clusters_silhouette(ax=axes[2], cmap=cmap)

        plt.tight_layout()
        plt.savefig(output_file)

        return self

    def _plot_clusters_image(self, ax, cmap, mirror=True):

        im = np.array(self.df_cluster_).reshape(
            self.w_obj_.pixel_, self.w_obj_.pixel_)

        if mirror:
            im_left = im[:, :int(self.w_obj_.pixel_ / 2)]
            im[:, int(self.w_obj_.pixel_ / 2):] = im_left[:, ::-1]

        ax.imshow(im, cmap=cmap)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    def _plot_clusters_tsne(self, ax, cmap):

        X = self.tsne_spatial_pixels_.T[0]
        Y = self.tsne_spatial_pixels_.T[1]

        ax.scatter(X, Y, cmap=cmap, c=self.clusters_, s=10,
                   vmin=0, vmax=max(self.clusters_))

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    def _plot_clusters_silhouette(self, ax, cmap):

        silhouette_vals = silhouette_samples(self.tsne_spatial_pixels_,
                                             self.clusters_,
                                             metric="euclidean")

        y_ax_lower, y_ax_upper = 0, 0
        yticks = []

        for cl in range(1, max(self.clusters_)+1):
            c_silhouette_vals = silhouette_vals[self.clusters_ == cl]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cmap(cl / max(self.clusters_))

            ax.barh(range(y_ax_lower, y_ax_upper),
                    c_silhouette_vals,
                    height=1.0,
                    edgecolor='none',
                    color=color)

            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)

        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_yticks(yticks)
        ax.set_yticklabels(range(1, max(self.clusters_)+1))
        ax.set_ylabel('Cluster')
        ax.set_xlabel('Silhouette coefficient')

    def _calc_tsne(self, **kwargs):
        """Calculats t-SNE for two directions

        Parameters
        ----------
        **kwargs :
            keyword arguments for sklearn.manifold.TSNE.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        df = self.spatial_.ix[self.r_obj_.variable_genes_,
                              self.w_obj_.pixel_name_embryo_]

        self.tsne_spatial_pixels_ = TSNE(**kwargs).fit_transform(
            1 - df.corr())
        self.tsne_spatial_genes_ = TSNE(**kwargs).fit_transform(
            1 - df.T.corr())

        return self

    def _plot_tsne(self, output_dir):
        """Plots the results of t-SNE

        Parameters
        ----------
        output_dir : str
            The PNG file is exported to this directory.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        output_file = os.path.join(output_dir, "tsne_spatial.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes = axes.flatten()
        plot_tsne(self.tsne_spatial_pixels_, ax=axes[0])
        plot_tsne(self.tsne_spatial_genes_, ax=axes[1])
        axes[0].set_title("Spatial gene expression\n(t-SNE: pixels)")
        axes[1].set_title("Spatial gene expression\n(t-SNE: genes)")
        plt.tight_layout()
        plt.savefig(output_file)

        return self

    def _plot_image(self, output_file, gene_list,
                    is_uid=False, conversion_table_file=None):

        if is_uid:
            df = pd.read_csv(conversion_table_file, sep="\t",
                             names=('uid', 'symbol'))
            uid_list = gene_list
            symbol_list = [df[df.uid == x].symbol.values[0]
                           for x in gene_list]
            ax_title = [str(x) + " : " + str(y) for x, y in
                        zip(symbol_list, uid_list)]
        else:
            ax_title = gene_list

        ncol = 8
        nrow = np.int(np.ceil(1.0 * len(gene_list) / ncol))

        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten()

        for i, gene in enumerate(gene_list):
            ind = self.genes_.index(gene)
            im = self.spatial_images_[ind]
            axes[i].imshow(im, cmap=plt.cm.jet)
            axes[i].axis('off')
            axes[i].set_title(ax_title[i], fontsize=10)

        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(output_file)


def _make_custum_cmap(cmap):

    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (1, 1, 1, 1.0)
    cmap_custom = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    return cmap_custom
