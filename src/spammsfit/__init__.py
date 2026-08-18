"""SPAMMSFit: spectral inference using SPAMMS models."""

from spammsfit.configuration import SpammsConfig
from spammsfit.core import SpammsFit
from spammsfit.fits.bayesian import BayesianFit
from spammsfit.fits.chisquare import ChiSquareFit
from spammsfit.fits.differential_evolution import DEFit
from spammsfit.parameters import (
    Parameter,
    ParameterSet,
)
from spammsfit.results.bayesian import (
    BayesianResult,
)
from spammsfit.results.chisquare import (
    ChiSquareResult,
)
from spammsfit.results.differential_evolution import (
    DEResult,
)
from spammsfit.spectrum import Spectrum
from spammsfit.utilities.grid_index import (
    create_model_index,
)


__version__ = "0.1.0"


__all__ = [
    "BayesianFit",
    "BayesianResult",
    "ChiSquareFit",
    "ChiSquareResult",
    "DEFit",
    "DEResult",
    "Parameter",
    "ParameterSet",
    "SpammsConfig",
    "SpammsFit",
    "Spectrum",
    "create_model_index",
]
