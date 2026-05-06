import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

def plot_snr_boxplot(snr_subject_values, methods, labels=None, save_path=None):
    data = [snr_subject_values[method] for method in methods]

    if labels is None:
        labels = methods

    plt.figure(figsize=(7, 4.8))

    plt.boxplot(
        data,
        labels=labels,
        showmeans=False,
        showfliers=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    plt.ylabel("LCModel-reported SNR", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylim(0, 30)

    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(pad=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()