from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import pandas as pd
import numpy as np


__all__ = ['Position', ]


class Position(object):
    """Class for position information

    Attributes
    ----------
    r_obj_ : RNAseq object
        Input RNAseq instance
    w_obj_ : Wish object
        Input Wish instance
    genes_ : list
        List of genes used for position inference.
    position_ : pandas DataFrame
        Inferred cell position
    position_images_ : list of ndarray
        list of inferred cell position
    mean_position_ : ndarray, shape (w_obj_.pixel_ ** 2, )
        Mean of inferred cell position.
    mean_position_image_ : ndarray, shape (w_obj_.pixel_, w_obj_.pixel_)
        Image of mean inferred cell position.
    position_loocv_ : list
        List of position matrix for LOOCV
    position_loocv_mean_ : list
        List of mean of position matrix for LOOCV
    """

    def __init__(self):
        pass

    def load_inputs(self, r_obj, w_obj):
        """ Loads RNA-seq object and WISH object.

        Parameters
        ----------
        r_obj : :obj:`RNAseq`
            Instance of RNA-seq class

        w_obj : :obj:`Wish`
            Instance of Wish class

        Return
        ------
        self : obj
            Returns the instance itself.

        """
        self.r_obj_ = r_obj
        self.w_obj_ = w_obj

        self.num_cells_ = self.r_obj_.num_cells_

        if self.r_obj_.annotation_type_ is not self.w_obj_.annotation_type_:
            print("Annotation types of genes are different in two matrix.")
            raise ValueError

        return self

    def infer_position(self):
        """ Calculates position of cells.

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance itself.

        """
        self.genes_ = list(set(self.r_obj_.genes_) &
                           set(self.w_obj_.genes_))

        self.position_, self.position_images_ = (
            self._calc_position(self.genes_))

        self.mean_position_, self.mean_position_image_ = (
            self._mean_position(self.position_))

        return self

    def calc_position_loocv(self):
        """Calculates position matrix for LOOCV

        LOOCV: Leave-one-out cross-validation
        The self.position_loocv_ and self.position_loocv_mean
        can be used for further calculation in primo.spatial_pattern

        Parameters
        ----------

        Return
        ------
        self : object
            Returns the instance itself.

        """

        self.position_loocv_ = []
        self.position_loocv_mean_ = []

        for i, gene in enumerate(self.genes_):
            genes_exclude = [x for x in self.genes_ if x != gene]

            position, _ = self._calc_position(genes_exclude)
            mean_position, _ = self._mean_position(position)

            self.position_loocv_.append(position)
            self.position_loocv_mean_.append(mean_position)

        return self

    def _calc_position(self, genes):
        """Calculates position matrix

        Parameters
        ----------
        genes : :obj:`list`
            list of genes

        Return
        ------
        position : :obj:`Dataframe`
            position matrix
        position_images : :obj:`list` of :obj:`ndarra`
            list of position matrix

        """
        r_mat = self.r_obj_.df_rnaseq_.ix[genes, :]

        # scaling-like operation for genes
        # to weaken high expression gene
        r_mat = (r_mat.T / r_mat.sum(axis=1)).T

        r_mat_norm = np.sqrt(np.square(r_mat).sum(axis=0))
        r_mat = (r_mat / r_mat_norm).fillna(0)

        w_mat = self.w_obj_.wish_matrix_.ix[genes, :]
        w_mat_norm = np.sqrt(np.square(w_mat).sum(axis=0))
        w_mat = (w_mat / w_mat_norm).fillna(0)

        cosine_similarity = np.dot(r_mat.T, w_mat)

        cosine_similarity_norm = (cosine_similarity.T /
                                  cosine_similarity.sum(axis=1)).T

        position = pd.DataFrame(cosine_similarity_norm,
                                index=self.r_obj_.cells_,
                                columns=self.w_obj_.wish_matrix_.columns)

        position_images = [np.array(position.ix[i]).reshape(
            self.w_obj_.pixel_, self.w_obj_.pixel_)
                                for i in range(position.shape[0])]

        return (position, position_images)

    def _mean_position(self, position):
        """Calculates meand of position matrix

        Parameters
        ----------
        position : :obj:`Dataframe`
            Position matrix, shape (n_pixel, )

        Return
        ------
        mean_position : :obj:`Series`
        mean_position_image : :obj:`list` of :obj:`ndarray`

        """
        mean_position = position.mean()
        mean_position_image = mean_position.reshape(
            self.w_obj_.pixel_, self.w_obj_.pixel_)

        return (mean_position, mean_position_image)

    def plot_position(self, output_dir, num_cells=None):
        """Plot inferred cell position

        Parameters
        ----------
        output_dir : str
            The png file is exported to this directory.
        num_cells : :obj:`int`, optional
            Number of cells to be shown in exported png file.
            If `None`, all cells will be shown.

        Return
        ------
        self : object
            Returns the instance itself.

        """

        if num_cells is None:
            num_cells = self.num_cells_

        output_file = os.path.join(output_dir, "cell_position.png")
        ncol = 8
        nrow = np.int(np.ceil(1.0 * num_cells / ncol))

        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten()

        for i, cell in enumerate(self.r_obj_.cells_[0: num_cells]):
            axes[i].imshow(self.position_images_[i], cmap=plt.cm.jet)
            axes[i].axis('off')
            axes[i].set_title(cell, fontsize=10)

        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(output_file)

        self._plot_mean_position(output_dir)

        return self

    def _plot_mean_position(self, output_dir):
        """Plot mean of inferred cell position

        Parameters
        ----------
        output_dir : str
            The png file is exported to this directory.

        Return
        ------
        self : object
            Returns the instance itself.

        """

        output_file = os.path.join(output_dir, "cell_position_mean.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(self.mean_position_image_)
        ax.axis('off')
        ax.set_title("Mean of cell position probability", fontsize=24)
        plt.tight_layout()
        plt.savefig(output_file)

        return self
