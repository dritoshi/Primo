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

        if self.r_obj_.annotation_type_ is not self.w_obj_.annotation_type_:
            print("Annotation types of genes are different in two matrix.")
            raise ValueError

        return self

    def calc_position(self):
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

        # writing
        r_mat = self.r_obj_.df_rnaseq_.ix[self.genes_, :]

        # scaling-like operation for genes
        # to weaken high expression gene
        r_mat = (r_mat.T / r_mat.sum(axis=1)).T

        r_mat_norm = np.sqrt(np.square(r_mat).sum(axis=0))
        r_mat = (r_mat / r_mat_norm).fillna(0)

        w_mat = self.w_obj_.wish_matrix_.ix[self.genes_, :]
        w_mat_norm = np.sqrt(np.square(w_mat).sum(axis=0))
        w_mat = (w_mat / w_mat_norm).fillna(0)

        cosine_similarity = np.dot(r_mat.T, w_mat)

        cosine_similarity_norm = (cosine_similarity.T /
                                  cosine_similarity.sum(axis=1)).T

        self.position_ = pd.DataFrame(cosine_similarity_norm,
                                      index=self.r_obj_.cells_,
                                      columns=self.w_obj_.wish_matrix_.columns)

        self.num_cells_ = self.position_.shape[0]
        self.num_pixels_ = self.position_.shape[1]

        self.position_images_ = [np.array(self.position_.ix[i]).reshape(
            self.w_obj_.pixel_, self.w_obj_.pixel_)
                                for i in range(self.num_cells_)]

        self.mean_position_ = self.position_.mean()
        self.mean_position_image_ = self.mean_position_.reshape(
            self.w_obj_.pixel_, self.w_obj_.pixel_)

        return self

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


if __name__ == '__main__':
    pass
