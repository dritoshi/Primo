from __future__ import print_function

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
