import numpy as np

import matplotlib.pyplot as plt
import mplhep as mh

mh.style.use("ATLAS")

class MaiaAxis:
    def __init__(self, outpath, **kwargs):
        fig, ax = plt.subplots()
        mh.label.exp_text(
            "Muon Collider", 
            "Simulation",
            "Internal",
            "MAIA Detector Concept",
            loc=1,
            fontstyle=("italic", "normal", "normal", "normal"),
            ax=ax
        )

        self.fig = fig
        self.ax = ax
        self.outpath = outpath
        self.kwargs = kwargs

    def __enter__(self):
        return self.ax

    def __exit__(self, type, value, traceback):
        self.fig.savefig(self.outpath, **self.kwargs)
        plt.close(self.fig)

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
    if density:
        ylabel += " Normalized"

    with MaiaAxis(outpath) as ax:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xbounds is not None:
            ax.set_xlim(*xbounds)
        if ybounds is not None:
            ax.set_ylim(*ybounds)
        if log_scale:
            ax.set_yscale("log")

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
    with MaiaAxis(outpath, bbox_inches="tight") as ax:
        mh.hist2dplot(
            histogram,
            ax=ax,
            mask=mask
        )

def maia_hist2d_residual(
    residual_histogram : tuple,
    outpath : str,
    xlabel : str,
    ylabel : str,
    xbounds : tuple | None = None,
    ybounds : tuple | None = None,
    cbar_label : str = "Gen - MC"
):
    """
    Plot a 2D residual (difference of two histograms, same binning) with a
    diverging colormap centered at zero, so excess/deficit regions read as
    opposite colors instead of sharing one sequential scale with the raw counts.
    """
    counts, _, _ = residual_histogram
    vmax = np.abs(counts).max()
    vmax = vmax if vmax > 0 else 1.0
    with MaiaAxis(outpath, bbox_inches="tight") as ax:
        artists = mh.hist2dplot(
            residual_histogram,
            ax=ax,
            cmap="RdBu_r",
            # cmin/cmax double as pcolormesh's vmin/vmax inside hist2dplot; passing vmin/vmax
            # directly in **kwargs would collide with that and raise a duplicate-keyword error.
            cmin=-vmax,
            cmax=vmax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xbounds is not None:
            ax.set_xlim(*xbounds)
        if ybounds is not None:
            ax.set_ylim(*ybounds)
        if artists.cbar is not None:
            artists.cbar.set_label(cbar_label)