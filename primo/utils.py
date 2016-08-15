
import matplotlib.pyplot as plt

__all__ = ['plot_tsne', ]


def plot_tsne(X, ax):
    """Plot the results of t-SNE

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The resulted matrix of t-SNE.

    ax : matplotlib axis
        xxx

    """
    ax.scatter(X.T[0], X.T[1], c='darkgray', s=5, edgecolors='None')
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
