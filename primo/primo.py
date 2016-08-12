from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from distutils.version import StrictVersion
# import glob

import pandas as pd
import numpy as np

# # from scipy.stats import xxx

# from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, RandomizedPCA
# from sklearn.cluster import AggromerativeClustering

from scipy.misc import imresize

from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, img_as_bool
from skimage.morphology import (disk, binary_closing, binary_opening,
                                binary_dilation, binary_erosion,
                                remove_small_objects)

# import seaborn as sns


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
        mean = self.df_rnaseq_.sum().mean()
        sd = self.df_rnaseq_.sum().std()
        index_not_outlier = abs(self.df_rnaseq_.sum() - mean) < (val * sd)

        self.df_rnaseq_ = self.df_rnaseq_.ix[:, index_not_outlier]
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


class Wish(object):
    """Container of WISH pattern

    Attributes
    ----------
    wish_images_ : skimage image collection
        This contains original WISH images.
    genes_ : list
        List of genes of which WISH patterns are imported.
    annotation_type_ : str
        Annotation type of genes. Examples: symbol or uid.
    genes_symbol_ : list
        Gene symbol of genes.
    genes_uid_ : list
        Unigene ID of genes.
    pixel : int
        The length of one side of filtered images.
    wish_images_filtered_ : list of ndarray
        List of images after filtered.
    wish_matrix_ : pandas DataFrame
        Wish matrix.
    """

    def __init__(self):
        pass

    def load_WISH_images(self, images_dir, annotation_type="symbol"):
        """Load image files of WISH pattern

        Parameters
        ----------
        images_dir : str
            PNG files are included in this directory.
            the file name should be gene symbol + .png.

        annotation_type : str
            Type of gene annotation.
            Examples: symbol, uid (Unigene ID)

        Return
        ------
        self : object
            Returns the instance itself.

        """
        png_path = os.path.join(images_dir, "*.png")
        self.wish_images_ = io.imread_collection(png_path)
        self.genes_ = [os.path.splitext(strings)[0].split("/")[-1]
                       for strings in self.wish_images_.files]

        self.annotation_type_ = annotation_type

        return self

    def symbol_to_uid(self, conversion_table_file):
        """Convert annotation type of gene from symbol to uid

        Parameters
        ----------
        conversion_table_file : str
            File path of conversion table, tab-separated text.
            Header should be 'symbol\tuid'

        Return
        ------
        self : object
            Returns the instance itself

        """

        df = pd.read_csv(conversion_table_file, sep="\t",
                         names=('uid', 'symbol'))

        self.genes_symbol_ = self.genes_
        self.genes_uid_ = [df[df.symbol == x].uid.values[0]
                           for x in self.genes_symbol_]
        self.genes_ = self.genes_uid_
        self.annotation_type_ = "uid"

        return self

    def filter_images(self, pixel):
        """filter images

        Parameters
        ----------
        pixel : int
            Target pixel size for resizing.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        self.pixel = pixel

        self.wish_images_filtered_ = [processing_image(im, self.pixel)
                                      for im in self.wish_images_]

        w_list = [x.flatten() for x in self.wish_images_filtered_]
        w_index = self.genes_
        w_column = ['pix' + str(i) for i in range(1, self.pixel ** 2 + 1)]
        self.wish_matrix_ = pd.DataFrame(w_list,
                                         index=w_index, columns=w_column)

        return self

    def plot_wish(self, output_dir):
        """Plot original and filtered WISH images

        Parameters
        ----------
        output_dir : str
            Image files are exported to output_dir

        Return
        ------
        self : object
            Returns the instance itself.

        """

        self._plot_wish_pattern(self.wish_images_)
        output_file = os.path.join(output_dir, "wish_original.png")
        plt.savefig(output_file)

        self._plot_wish_pattern(self.wish_images_filtered_)
        output_file = os.path.join(output_dir, "wish_filtered.png")
        plt.savefig(output_file)

        return self

    def _plot_wish_pattern(self, images):
        """plot images for WISH patterns

        Parameters
        ----------
        images: list
            list object of images
        """

        num_genes = len(self.genes_)
        ncol = 8
        nrow = np.int(np.ceil(1.0 * num_genes / ncol))

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten()

        for i, gene in enumerate(self.genes_):
            axes[i].imshow(images[i], cmap=plt.cm.Purples)
            axes[i].axis('off')
            axes[i].set_title(gene, fontsize=20)

        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()


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

    """

    def __init__(self):
        pass

    def load_inputs(self, r_obj, w_obj):
        """ Loads RNA-seq object and WISH object.

        Parameters
        ----------
        r_obj : obj: `RNAseq class object`
            Instance of RNA-seq class

        w_obj : obj: `Wish class object`
            Instance of Wishclass

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
        r_mat_norm = np.sqrt(np.square(r_mat).sum(axis=1))
        r_mat = (r_mat.T / r_mat_norm).T.fillna(0)

        w_mat = self.w_obj_.wish_matrix_.ix[self.genes_, :]
        w_mat_norm = np.sqrt(np.square(w_mat).sum(axis=1))
        w_mat = (w_mat.T / w_mat_norm).T.fillna(0)

        cosine_similarity = np.dot(r_mat.T, w_mat)

        cosine_similarity_norm = (cosine_similarity.T /
                                  cosine_similarity.sum(axis=1)).T

        self.position_ = pd.DataFrame(cosine_similarity_norm,
                                      index=self.r_obj_.cells_,
                                      columns=self.w_obj_.wish_matrix_.columns)

        self.num_cells_ = self.position_.shape[0]
        self.num_pixels_ = self.position_.shape[1]

        self.position_images_ = [np.array(self.position_.ix[i]).reshape(
            self.w_obj_.pixel, self.w_obj_.pixel)
                                for i in range(self.num_cells_)]

        return self

    def plot_position(self, output_dir, num_cells=None):
        """Plot inferred cell position

        Parameters
        ----------
        output_dir : str
            The png file is exported to this directory.
        num_cells : obj:‘`int`, optional
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

        plt.savefig(output_file)
        plt.tight_layout()

        return self


def _tsne(X, **kwargs):
    """t-SNE

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Gene expression matrix.
    **kwargs
        Arbitary keyword arguments.

    Return
    ------
    X_new: array-like, shape (n_samples, 2)

    """
    X_new = TSNE(**kwargs).fit_transform(X)
    return X_new
    # Is this really needed?


def plot_tsne(X, ax):
    """Plot the results of t-SNE

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The resulted matrix of t-SNE.

    ax : matplotlib axis
        xxx

    """
    ax.scatter(X.T[0], X.T[1], c='lightgray', s=5, edgecolors='None')
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)


def processing_image(image, pixel):
    """Preprocessing image

    Parameters
    ----------
    image : ndarray
        Binary input image.
    pixel : int
        Target pixel for resizing.

    Return
    ------
    image_out : ndarray, shape (pixel, pixel)
        Output image.

    """
    im = image
    im = imresize(im, (pixel, pixel), interp='bilinear')
    im = (rgb2gray(im) < 0.5)
    im = remove_small_objects(im, pixel / 6.)
    im = (binary_erosion(im, disk(pixel / 16.)) * 0.25 +
          im * 0.5 +
          binary_dilation(im, disk(pixel / 16.)) * 0.25)
    im = im * disk((pixel - 1) * 0.5)
    image_out = im

    return image_out


if __name__ == '__main__':
    pass
