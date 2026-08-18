"""Plotting utilities for SPAMMSFit."""

from spammsfit.plotting.bayesian import (
    plot_acceptance_fraction,
    plot_autocorrelation_time,
    plot_corner,
    plot_posterior_histograms,
    plot_trace,
)
from spammsfit.plotting.chisquare import (
    plot_chi2_corner,
    plot_parameter_map,
    plot_profile_likelihood,
)
from spammsfit.plotting.common import (
    plot_fit,
    plot_fit_with_residuals,
    plot_residuals,
)
from spammsfit.plotting.differential_evolution import (
    plot_convergence,
    plot_parameter_evolution,
    plot_population,
    plot_population_corner,
)


__all__ = [
    "plot_acceptance_fraction",
    "plot_autocorrelation_time",
    "plot_chi2_corner",
    "plot_convergence",
    "plot_corner",
    "plot_fit",
    "plot_fit_with_residuals",
    "plot_parameter_evolution",
    "plot_parameter_map",
    "plot_population",
    "plot_population_corner",
    "plot_posterior_histograms",
    "plot_profile_likelihood",
    "plot_residuals",
    "plot_trace",
]
