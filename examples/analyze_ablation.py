"""
Consolidated analysis for encoding/energy-parameterization ablations:
per-variable Wasserstein bar plots, grouped log-scale comparison, and
pairwise + aggregate Pareto fronts.

Reads directly from the outputs of plot_comparison.py -- no hardcoded
tags or paths. Run after generating + comparing all ablation variants:

    python analyze_ablation.py --plots-dir plots --history-dir training/history -o plots/analysis
"""
import argparse
import glob
import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    HAVE_ADJUSTTEXT = True
except ImportError:
    HAVE_ADJUSTTEXT = False

LABEL_MAP = {
    "energy": "Energy [GeV]",
    "phi": r"$\phi$",
    "eta": r"$\eta$",
    "s": "s [mm]",
    "z": "z [mm]",
}


def load_wasserstein(plots_dir):
    """Concatenate every wasserstein_distances.csv found under plots_dir. Tag comes from the file's own 'tag' column."""
    files = glob.glob(os.path.join(plots_dir, "**", "wasserstein_distances.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No wasserstein_distances.csv files found under {plots_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_val_loss(history_dir):
    """Best (min) val_loss per tag, tag inferred from 'history_<tag>.csv' filename."""
    files = glob.glob(os.path.join(history_dir, "**", "history_*.csv"), recursive=True)
    best_loss = {}
    for f in files:
        m = re.match(r"^history_(.+)\.csv$", os.path.basename(f))
        if not m:
            continue
        tag = m.group(1)
        df = pd.read_csv(f)
        best_loss[tag] = df["val_loss"].min()
    return pd.Series(best_loss)


def default_tag_sort_key(tag):
    """Group by prefix, rawE before logE, if the tag follows that convention; else plain alphabetical."""
    m = re.match(r"^(.*)_(rawE|logE)$", tag)
    if m:
        prefix, energy = m.groups()
        return (prefix, 0 if energy == "rawE" else 1)
    return (tag, -1)


def energy_param_of(tag):
    if tag.endswith("_logE"):
        return "logE"
    if tag.endswith("_rawE"):
        return "rawE"
    return "unknown"


def variance_mode_of(tag):
    """predict_variances=True runs are conventionally tagged with a '_predvar' token."""
    return "learned_var" if re.search(r"(?:^|_)predvar(?:$|_)", tag) else "fixed_var"


def resolve_tags(available, requested):
    if requested is None:
        return sorted(available, key=default_tag_sort_key)
    missing = [t for t in requested if t not in available]
    if missing:
        warnings.warn(f"Requested tags not found in data, dropping: {missing}")
    return [t for t in requested if t in available]


def pareto_mask(x, y):
    """True where a point is not strictly dominated (lower-is-better on both axes)."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False
                break
    return mask


def get_colors(tags):
    cmap = plt.cm.tab10.colors if len(tags) <= 10 else plt.cm.tab20.colors
    return dict(zip(tags, cmap))


def make_legend_handles(tags, colors):
    tag_handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[t], markersize=9, label=t) for t in tags]
    front_handle = plt.Line2D([0], [0], marker="o", linestyle="", color="gray",
                              markeredgecolor="black", markeredgewidth=1.5, markersize=9, label="Pareto optimal")
    return tag_handles + [front_handle]


def plot_front(ax, x, y, tags, colors, xlabel, ylabel):
    mask = pareto_mask(x.values, y.values)
    for i, tag in enumerate(tags):
        ax.scatter(x[tag], y[tag], color=colors[tag], s=90,
                   edgecolor="black" if mask[i] else "none", linewidth=1.5, zorder=3)
    front_idx = np.where(mask)[0]
    front_pts = sorted(zip(x.values[front_idx], y.values[front_idx]))
    if len(front_pts) > 1:
        fx, fy = zip(*front_pts)
        ax.plot(fx, fy, linestyle="--", color="gray", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mask


def label_points(ax, x, y, tags):
    if HAVE_ADJUSTTEXT:
        texts = [ax.text(x[t], y[t], t, fontsize=9) for t in tags]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8), expand=(1.3, 1.6))
    else:
        for t in tags:
            ax.annotate(t, (x[t], y[t]), fontsize=8, xytext=(5, 5), textcoords="offset points")


def main(args):
    os.makedirs(args.out, exist_ok=True)

    w_data = load_wasserstein(args.plots_dir)
    variables = w_data["variable"].unique().tolist()
    available_tags = w_data["tag"].unique().tolist()
    tags = resolve_tags(available_tags, args.tag_order.split(",") if args.tag_order else None)
    colors = get_colors(tags)
    pivot = w_data.pivot(index="tag", columns="variable", values="wasserstein_distance").reindex(tags)[variables]

    # ---- Plot 1: per-variable bar grid ----
    ncols = 3
    nrows = -(-len(variables) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, var in zip(axes, variables):
        ax.bar(tags, pivot[var], color=[colors[t] for t in tags])
        ax.set_title(f"Wasserstein Distance: {LABEL_MAP.get(var, var)}")
        ax.set_ylabel("Wasserstein Distance")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[len(variables):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "wasserstein_by_variable.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 2: grouped log-scale comparison ----
    x = np.arange(len(variables))
    width = 0.8 / len(tags)
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(variables) * len(tags)), 6))
    for i, tag in enumerate(tags):
        ax.bar(x + i * width - 0.4 + width / 2, pivot.loc[tag], width, label=tag, color=colors[tag])
    ax.set_yscale("log")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Wasserstein Distance (log scale)")
    ax.set_title("Wasserstein Distance by Variable and Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP.get(v, v) for v in variables])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "wasserstein_grouped.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 3: aggregate Wasserstein vs val loss, split by energy parameterization ----
    try:
        best_loss = load_val_loss(args.history_dir)
        norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())
        agg_score = norm.sum(axis=1)

        energy_groups = {t: energy_param_of(t) for t in tags}
        variance_groups = {t: variance_mode_of(t) for t in tags}

        # Only facet on energy parameterization if every tag is unambiguously marked --
        # mixing "unknown" with a known value would silently lump possibly-incomparable
        # tags together. Variance mode has no such "unknown" case: every tag is either
        # marked '_predvar' or defaults to fixed-variance, so it's always safe to facet on.
        energy_varies = len(set(energy_groups.values())) > 1 and "unknown" not in energy_groups.values()
        variance_varies = len(set(variance_groups.values())) > 1

        def regime_of(t):
            parts = []
            if energy_varies:
                parts.append(energy_groups[t])
            if variance_varies:
                parts.append(variance_groups[t])
            return tuple(parts)

        groups = {t: regime_of(t) for t in tags}
        regime_values = sorted(set(groups.values()))
        incomparable_axes = (["energy parameterization"] if energy_varies else []) + \
                             (["variance mode"] if variance_varies else [])

        if regime_values and regime_values != [()]:
            fig, axes = plt.subplots(1, len(regime_values), figsize=(7 * len(regime_values), 6))
            axes = np.atleast_1d(axes)
            for ax, regime in zip(axes, regime_values):
                sub_tags = [t for t in tags if groups[t] == regime and t in best_loss.index]
                if not sub_tags:
                    ax.axis("off")
                    continue
                plot_front(ax, agg_score[sub_tags], best_loss[sub_tags], sub_tags, colors,
                           "Aggregate Wasserstein Score (normalized sum)", "Best Validation Loss")
                label_points(ax, agg_score, best_loss, sub_tags)
                ax.set_title("{} runs only\n(val loss not comparable across {})".format(
                    " / ".join(regime), " or ".join(incomparable_axes)))
            fig.suptitle("Pareto Front: Distribution Fidelity vs Training Performance", fontsize=13)
        else:
            valid_tags = [t for t in tags if t in best_loss.index]
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_front(ax, agg_score[valid_tags], best_loss[valid_tags], valid_tags, colors,
                       "Aggregate Wasserstein Score (normalized sum)", "Best Validation Loss")
            label_points(ax, agg_score, best_loss, valid_tags)
            ax.set_title("Pareto Front: Distribution Fidelity vs Training Performance")

        fig.legend(handles=make_legend_handles(tags, colors), loc="lower center", ncol=min(len(tags), 7), fontsize=8)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(os.path.join(args.out, "pareto_aggregate_vs_valloss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except FileNotFoundError:
        print(f"No history CSVs found under {args.history_dir} -- skipping aggregate-vs-val-loss plot.")

    # ---- Plot 4: all pairwise Wasserstein-vs-Wasserstein Pareto fronts (unaffected by the val-loss issue) ----
    pairs = list(combinations(variables, 2))
    ncols = min(5, len(pairs))
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (v1, v2) in zip(axes, pairs):
        plot_front(ax, pivot[v1], pivot[v2], tags, colors, LABEL_MAP.get(v1, v1), LABEL_MAP.get(v2, v2))
        ax.set_title(f"{LABEL_MAP.get(v1, v1)} vs {LABEL_MAP.get(v2, v2)}", fontsize=10)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.legend(handles=make_legend_handles(tags, colors), loc="lower center", ncol=min(len(tags) + 1, 7), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pairwise Pareto Fronts Across All Wasserstein Distance Combinations", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(args.out, "pareto_pairwise_all_combos.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote 4 plots to {args.out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze encoding/energy-parameterization ablation results")
    parser.add_argument("--plots-dir", default="plots", help="Directory containing per-tag wasserstein_distances.csv files (searched recursively)")
    parser.add_argument("--history-dir", default="training/history", help="Directory containing history_<tag>.csv files (searched recursively)")
    parser.add_argument("-o", "--out", default="plots/analysis", help="Output directory for generated plots")
    parser.add_argument("--tag-order", default=None, help="Comma-separated explicit tag order/subset (default: auto-sorted from available data)")
    print("\nFinished with exit code:", main(parser.parse_args()))
