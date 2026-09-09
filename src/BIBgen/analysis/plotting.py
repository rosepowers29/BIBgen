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