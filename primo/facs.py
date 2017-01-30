from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from distutils.version import StrictVersion

import pandas as pd
import numpy as np
from scipy.stats import entropy

import seaborn as sns
sns.set_style("white")

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__all__ = ['FACS', ]


class FACS(object):
    """Class for FACS data

    explanations

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self):
        pass

    def load_FACSdata(self, path, machine):
        """Load FACS file

        Parameters
        ----------
        path : str
            File path
        machine : str
            Name of FACS machine. Ex. "MoFlo", "SH-800"

        Return
        ------
        self : object
            Returns the instance itself
        """

        if machine == "MoFlo":
            df = pd.read_csv(path, sep=",", header=None)
            self.df_facs_info_ = df.ix[0:9, 0:1]
            df = pd.read_csv(path, sep=",").ix[:, 3:]

        elif machine == "SH-800":
            df = pd.read_csv(path, sep=",")
            df["Sort Index X"] = [x[1:].zfill(2) for x in df["Index"]]
            df["Sort Index Y"] = [x[0] for x in df["Index"]]
        else:
            print('Currently, machine must be "MoFlo" or "SH-800".')

        df["well"] = [str(x) + str(y).zfill(2) for x, y in zip(
            df["Sort Index X"], df["Sort Index Y"])]

        self.df_facs_data_ = df

        return self

    def load_cell_barcode_info(self, path):
        """Load cell barcode information file

        Parameters
        ----------
        path : str
            File path

        Return
        ------
        self : object
            Returns the instance itself
        """

        barcode_well = pd.read_csv(path, sep="\t")
        self.df_barcode_facs_ = pd.merge(barcode_well, self.df_facs_data_,
                                         on="well")

        return self
