"""Fitting-result containers provided by SPAMMSFit."""

from spammsfit.results.bayesian import (
    BayesianResult,
)
from spammsfit.results.chisquare import (
    ChiSquareResult,
)
from spammsfit.results.differential_evolution import (
    DEResult,
)


__all__ = [
    "BayesianResult",
    "ChiSquareResult",
    "DEResult",
]
