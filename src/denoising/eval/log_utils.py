import re
import numpy as np

def parse_training_log(log_path):
    epochs = []
    train_losses = []
    val_losses = []

    # Regex für die relevanten Zeilen
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+·\s+train=([0-9.eE+-]+)\s+·\s+val=([0-9.eE+-]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    # als numpy array (N, 3): [epoch, train, val]
    data = np.stack([epochs, train_losses, val_losses], axis=1)
    return data