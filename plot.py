#!/usr/local/Caskroom/miniconda/base/bin/python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Args, check_folder


def get_args():
    parser = argparse.ArgumentParser("Plot charts")
    parser.add_argument("path", type=str, help="Path to a Python file containing the data to plot")
    args = parser.parse_args()
    return args


def plot_results(
    records: dict,
    nvec: np.ndarray,
    dvec: np.ndarray,
    extend=None,
    cmap=Args.cmap,
    show_boundary=False,
    save_folder: str = None,
):
    """
    Plot the results of the experiments.
    If plotting phase transition for distance, use extend='max'
    """
    xgrid, ygrid = np.meshgrid(nvec, dvec)
    fig, (axes,) = plt.subplots(1, len(records), figsize=(5 * len(records), 5), squeeze=False)
    for ax, (prop, values) in zip(axes, records.items()):
        ax.set_title(prop)
        ax.set_xlabel("n")
        ax.set_ylabel("d")
        ax.set_xlim(nvec[0], nvec[-1])
        ax.set_ylim(dvec[0], dvec[-1])
        # ax.set_xticks(nvec)
        # ax.set_yticks(dvec)

        grid = np.mean(values, axis=2).T
        cs = ax.contourf(xgrid, ygrid, grid, levels=np.linspace(0, 1, 11), cmap=cmap, extend=extend)
        if show_boundary:
            ax.contour(nvec, dvec, grid, levels=(0.9, 1), colors=("k",), linewidths=(2,))
        fig.colorbar(cs, ax=ax)

    if save_folder is not None:
        check_folder(save_folder)
        path = save_folder + "figure.png"
        print("Saving figure to " + path)
        fig.savefig(path)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    path = Path(args.path)
    data = np.load(path)
    maxn, maxd, _ = data.shape
    plot_results({path.name: data}, np.arange(1, maxn + 1) * 10, np.arange(1, maxd + 1) * 10)
