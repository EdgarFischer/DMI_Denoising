import os
import re
import numpy as np


def parse_training_logs(log_paths=None, run_dirs=None, base_dir=None, log_name="train.log"):
    """
    Parse one or multiple training log files.

    Parameters
    ----------
    log_paths : list[str] or None
        Full paths to log files.
    run_dirs : list[str] or None
        List of run directory names. Combined with base_dir and log_name.
    base_dir : str or None
        Base directory containing the run directories.
    log_name : str
        Name of the log file inside each run directory.

    Returns
    -------
    curves : list[np.ndarray]
        List of arrays with shape (N, 3), columns:
        [epoch, train_loss, val_loss]
    """
    if log_paths is None:
        if run_dirs is None or base_dir is None:
            raise ValueError("Provide either log_paths or run_dirs together with base_dir.")
        log_paths = [os.path.join(base_dir, run_dir, log_name) for run_dir in run_dirs]

    pattern = re.compile(
        r"Epoch\s+(\d+)\s+·\s+train=([0-9.eE+-]+)\s+·\s+val=([0-9.eE+-]+)"
    )

    curves = []

    for log_path in log_paths:
        epochs = []
        train_losses = []
        val_losses = []

        with open(log_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    val_losses.append(float(match.group(3)))

        if len(epochs) == 0:
            raise ValueError(f"No training curve entries found in log file: {log_path}")

        curve = np.stack([epochs, train_losses, val_losses], axis=1)
        curves.append(curve)

    return curves