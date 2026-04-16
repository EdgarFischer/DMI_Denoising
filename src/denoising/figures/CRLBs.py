import matplotlib.pylab as plt

def plot_sd_boxplots_threshold(data, labels, stats, threshold=20, log_scale=True, showfliers=True, save_path=None):
    plt.figure(figsize=(12,6))

    bp = plt.boxplot(
        data,
        tick_labels=labels,
        showfliers=showfliers,   
        patch_artist=True
    )

    # Boxen einfärben
    for i, box in enumerate(bp['boxes']):
        metab = labels[i]
        median = stats[metab]["median"]

        if median > threshold:
            box.set_facecolor("red")
        else:
            box.set_facecolor("lightgray")

    # Median-Linien
    for median_line in bp['medians']:
        median_line.set_color("black")

    if log_scale:
        plt.yscale("log")

    plt.axhline(threshold, linestyle="--", color="red", label=f"{threshold}% threshold")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("CRLB (%)")
    plt.title("CRLB distribution per metabolite")

    plt.tight_layout()

    if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {save_path}")

    plt.show()