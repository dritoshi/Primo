from __future__ import print_function

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# import pandas as pd
import numpy as np
import pandas as pd

__all__ = ['SpatialExpression', ]


class SpatialExpression(object):
    """Class for spatial gene expression pattern

    Attributes
    ----------
    r_obj_ : :obj:`RNAseq`
        Instance of `RNAseq` class instance from primo.rnaseq.
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

    def load_inputs(self, r_obj, p_obj):
        """Load input objects

        Parameters
        ----------
        r_obj : :obj:`RNAseq`.
            primo RNAseq class instance.
        p_obj : :obj:`Position`.
            primo Position class instance.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        self.r_obj_ = r_obj
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

        self.spatial_ = np.dot(X, Y)
        self.spatial_ = self.spatial_ / np.array(self.p_obj_.mean_position_)
        self.spatial_ = pd.DataFrame(self.spatial_,
                                     index=self.genes_,
                                     columns=self.pixel_names_).fillna(0)

        self.spatial_images_ = [np.array(self.spatial_.ix[i]).reshape(
            self.pixel_, self.pixel_)
                                for i in range(len(self.genes_))]

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
            spatial_image = spatial.reshape(self.pixel_, self.pixel_)

            self.spatial_loocv_.append(spatial)
            self.spatial_loocv_images_.append(spatial_image)

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
