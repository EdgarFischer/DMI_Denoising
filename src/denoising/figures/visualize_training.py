import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("../src"))

from denoising.eval.log_utils import parse_training_logs


def plot_training_curves(
    run_dirs,
    base_dir,
    log_name="logs/train.log",
    show_train=True,
    show_val=True,
    log_scale=True,
):
    curves = parse_training_logs(
        run_dirs=run_dirs,
        base_dir=base_dir,
        log_name=log_name,
    )

    for run_dir, data in zip(run_dirs, curves):
        epochs = data[:, 0]
        train = data[:, 1]
        val = data[:, 2]

        if show_train:
            plt.plot(epochs, train, label=f"{run_dir} train")

        if show_val:
            plt.plot(epochs, val, label=f"{run_dir} val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.show()