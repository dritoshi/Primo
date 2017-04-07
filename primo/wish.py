from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import pandas as pd
import numpy as np

from scipy.misc import imresize

from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import (disk, binary_dilation, binary_erosion,
                                remove_small_objects)

__all__ = ['Wish', 'processing_image']


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
    pixel_ : int
        The length of one side of filtered images.
    wish_images_filtered_ : list of ndarray
        List of images after filtered.
    wish_matrix_ : pandas DataFrame
        Wish matrix.
    pixel_name_all_ : list
        list of pixel name for all pixels
    pixel_name_embryo_ : list
        list of pixel name for pixels located in embryo
    pixel_name_outer_ : list
        list of pixel name for pixels locates outer embryo
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

    def filter_images(self, pixel_v, pixel_h, shape="round"):
        """filter images

        Parameters
        ----------
        pixel_v : int
            Target pixel for resizing. Vertical.
        pixel_h : int
            Target pixel for resizing. Horizontal.

        Return
        ------
        self : object
            Returns the instance itself.

        """
        self.pixel_ = np.int(np.sqrt(pixel_v * pixel_h))
        self.pixel_v_ = pixel_v
        self.pixel_h_ = pixel_h

        self.wish_images_filtered_ = [processing_image(im, pixel_v, pixel_h, shape)
                                      for im in self.wish_images_]

        w_list = [x.flatten() for x in self.wish_images_filtered_]
        w_index = self.genes_
        w_column = ['pix' + str(i) for i in range(1, pixel_v * pixel_h + 1)]
        self.wish_matrix_ = pd.DataFrame(w_list,
                                         index=w_index, columns=w_column)

        if shape == "round":
            disk_matrix = disk((self.pixel_ - 1) * 0.5)
            self.pixel_name_all_ = np.array(w_column)
            self.pixel_name_embryo_ = np.array(w_column)[
                disk_matrix.flatten() == 1]
            self.pixel_name_outer_ = np.array(w_column)[
                disk_matrix.flatten() == 0]
        else:
            self.pixel_name_all_ = np.array(w_column)
            self.pixel_name_embryo_ = np.array(w_column)
            self.pixel_name_outer_ = np.array([])

        return self

    def plot_wish(self, output_dir, cmap=plt.cm.Purples):
        """Plot original and filtered WISH images

        Parameters
        ----------
        output_dir : str
            Image files are exported to output_dir
        cmap : :obj:`matplotlib.colors.Colormap`, optional, default: plt.cm.Purples
            Matplotlib color map

        Return
        ------
        self : object
            Returns the instance itself.

        """

        self._plot_wish_pattern(self.wish_images_, cmap)
        output_file = os.path.join(output_dir, "wish_original.png")
        plt.savefig(output_file)

        self._plot_wish_pattern(self.wish_images_filtered_, cmap)
        output_file = os.path.join(output_dir, "wish_filtered.png")
        plt.savefig(output_file)

        return self

    def _plot_wish_pattern(self, images, cmap):
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
            axes[i].imshow(images[i], cmap=cmap)
            axes[i].axis('off')
            axes[i].set_title(gene, fontsize=20)

        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)

        plt.tight_layout()


def processing_image(image, pixel_v, pixel_h, shape="round"):
    """Preprocessing image

    Parameters
    ----------
    image : ndarray
        Binary input image.
    pixel_v : int
        Target pixel for resizing. Vertical.
    pixel_h : int
        Target pixel for resizing. Horizontal.

    Return
    ------
    image_out : ndarray, shape (pixel_v, pixel_h)
        Output image.

    """

    pixel = np.int(np.sqrt(pixel_v * pixel_h))

    im = image
    im = imresize(im, (pixel_v, pixel_h), interp='bilinear')
    im = (rgb2gray(im) < 0.5)
    im = remove_small_objects(im, pixel / 6.)
    im = (binary_erosion(im, disk(pixel / 16.)) * 0.25 +
          im * 0.5 +
          binary_dilation(im, disk(pixel / 16.)) * 0.25)
    if shape == "round":
        im = im * disk((pixel - 1) * 0.5)
    image_out = im

    return image_out


if __name__ == '__main__':
    pass
