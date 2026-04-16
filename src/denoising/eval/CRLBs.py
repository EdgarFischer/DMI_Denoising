import numpy as np

def summarize_sd_maps(sd_maps):
    """
    Berechnet Median und IQR für SD/CRLB maps.
    Ignoriert NaNs und 0-Werte.

    Parameters
    ----------
    sd_maps : dict
        Dict mit {metabolite: np.ndarray}

    Returns
    -------
    summary : dict
        Dict mit Statistik pro Metabolit
    """
    summary = {}

    for metab, arr in sd_maps.items():
        vals = np.asarray(arr).ravel()

        n_total = vals.size
        n_nan = np.isnan(vals).sum()
        n_zero = np.sum(vals == 0)

        # gültige Werte
        valid = vals[np.isfinite(vals)]
        valid = valid[valid > 0]

        if valid.size == 0:
            summary[metab] = {
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "iqr": np.nan,
                "n_valid": 0,
                "n_total": n_total,
                "perc_nan": 100 * n_nan / n_total,
                "perc_zero": 100 * n_zero / n_total,
            }
            continue

        q25 = np.percentile(valid, 25)
        q75 = np.percentile(valid, 75)

        summary[metab] = {
            "median": np.median(valid),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "n_valid": valid.size,
            "n_total": n_total,
            "perc_nan": 100 * n_nan / n_total,
            "perc_zero": 100 * n_zero / n_total,
        }

    return summary

def prepare_boxplot_data(sd_maps, ignore_zeros=True, ignore_nans=True, sort_by_median=False):
    """
    Bereitet SD/CRLB maps für Boxplots vor.

    Parameters
    ----------
    sd_maps : dict
        Dict der Form {metabolite: np.ndarray}
    ignore_zeros : bool
        Ob 0-Werte entfernt werden sollen.
    ignore_nans : bool
        Ob NaN/Inf entfernt werden sollen.
    sort_by_median : bool
        Ob Metabolite nach Median sortiert werden sollen.

    Returns
    -------
    data : list of np.ndarray
        Liste mit validen Werten pro Metabolit, geeignet für plt.boxplot(data)
    labels : list of str
        Metabolitnamen in derselben Reihenfolge wie data
    stats : dict
        Kleine Zusatzstatistik pro Metabolit
    """
    data = []
    labels = []
    stats = {}

    for metab, arr in sd_maps.items():
        vals = np.asarray(arr).ravel()

        if ignore_nans:
            vals = vals[np.isfinite(vals)]

        if ignore_zeros:
            vals = vals[vals > 0]

        if vals.size == 0:
            continue

        data.append(vals)
        labels.append(metab)
        stats[metab] = {
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "n": int(vals.size),
        }

    if sort_by_median:
        order = np.argsort([stats[label]["median"] for label in labels])
        data = [data[i] for i in order]
        labels = [labels[i] for i in order]

    return data, labels, stats