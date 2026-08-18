"""Fitting methods provided by SPAMMSFit."""

from spammsfit.fits.bayesian import BayesianFit
from spammsfit.fits.chisquare import ChiSquareFit
from spammsfit.fits.differential_evolution import DEFit


__all__ = [
    "BayesianFit",
    "ChiSquareFit",
    "DEFit",
]
