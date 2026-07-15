import numpy as np

import matplotlib.pyplot as plt
import mplhep as mh

mh.style.use("ATLAS")

def maia_plot(
    xlabel : str,
    ylabel : str,
    xbounds : tuple | None = None,
    ybounds : tuple | None = None,
    log_scale : bool = False,
):
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xbounds is not None:
        ax.set_xlim(*xbounds)
    if ybounds is not None:
        ax.set_ylim(*ybounds)
    if log_scale:
        ax.set_yscale("log")

    mh.label.exp_text(
        "Muon Collider", 
        "Simulation",
        "Internal",
        "MAIA Detector Concept",
        loc=1,
        fontstyle=("italic", "normal", "normal", "normal"),
        ax=ax
    )
    return fig, ax

def maia_hist1d(
    histograms : dict[str,tuple],
    outpath : str,
    xlabel : str,
    ylabel : str = "Events",
    density : bool = False,
    xbounds : tuple | None = None,
    ybounds : tuple | None = None,
    log_scale : bool = False
):
    fig, ax = maia_plot(xlabel, ylabel, xbounds, ybounds, log_scale)
    for ids, ds in enumerate(histograms):
        mh.histplot(
            histograms[ds],
            histtype="step",
            ax=ax,
            label=ds,
            color="C{}".format(ids),
            density=density,
        )

    ax.legend()
    plt.savefig(outpath)
    plt.close(fig)

def maia_hist2d(
    histogram : tuple,
    outpath : str,
    xlabel : str,
    ylabel : str,
    xbounds : tuple | None = None,
    ybounds : tuple | None = None,
    mask_zero : bool = False
):
    mask = histogram[0] > 0 if mask_zero else None
    fig, ax = maia_plot(xlabel, ylabel, xbounds, ybounds)
    mh.hist2dplot(
        histogram,
        ax=ax,
        mask=mask
    )

    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)