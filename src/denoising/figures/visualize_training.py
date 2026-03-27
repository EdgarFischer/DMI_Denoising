import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("../src"))

from denoising.eval.log_utils import parse_training_log


def plot_training_curve(log_path, log_scale=True):
    data = parse_training_log(log_path)

    epochs = data[:, 0]
    train = data[:, 1]
    val = data[:, 2]

    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.show()