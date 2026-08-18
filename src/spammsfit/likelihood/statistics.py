"""Statistical calculations used by SPAMMSFit."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


LOG_TWO_PI = np.log(2.0 * np.pi)


def residuals(
    observed_flux: ArrayLike,
    model_flux: ArrayLike,
    *,
    validate: bool = False,
) -> FloatArray:
    """
    Calculate observed-minus-model residuals.

    Parameters
    ----------
    observed_flux
        Observed flux values.
    model_flux
        Model flux values evaluated at the observed wavelengths.
    validate
        Validate array shapes and finite values when True.

    Returns
    -------
    numpy.ndarray
        Residual array defined as observed flux minus model flux.
    """

    observed = np.asarray(
        observed_flux,
        dtype=np.float64,
    )

    model = np.asarray(
        model_flux,
        dtype=np.float64,
    )

    if validate:
        _validate_flux_arrays(
            observed_flux=observed,
            model_flux=model,
        )

    return observed - model


def normalized_residuals(
    observed_flux: ArrayLike,
    model_flux: ArrayLike,
    uncertainty: ArrayLike | float,
    *,
    validate: bool = False,
) -> FloatArray:
    """
    Calculate residuals normalized by observational uncertainty.
    """

    observed = np.asarray(
        observed_flux,
        dtype=np.float64,
    )

    model = np.asarray(
        model_flux,
        dtype=np.float64,
    )

    error = np.asarray(
        uncertainty,
        dtype=np.float64,
    )

    if validate:
        _validate_statistical_arrays(
            observed_flux=observed,
            model_flux=model,
            uncertainty=error,
        )

    return (observed - model) / error


def chi_square(
    observed_flux: ArrayLike,
    model_flux: ArrayLike,
    uncertainty: ArrayLike | float,
    *,
    validate: bool = False,
) -> float:
    """
    Calculate chi-square.

    The definition is

    chi2 = sum(((observed - model) / uncertainty)**2).
    """

    normalized = normalized_residuals(
        observed_flux=observed_flux,
        model_flux=model_flux,
        uncertainty=uncertainty,
        validate=validate,
    )

    return float(
        np.dot(
            normalized,
            normalized,
        )
    )


def gaussian_log_likelihood(
    observed_flux: ArrayLike,
    model_flux: ArrayLike,
    uncertainty: ArrayLike | float,
    *,
    include_normalization: bool = True,
    validate: bool = False,
) -> float:
    """
    Calculate the independent Gaussian log likelihood.

    Parameters
    ----------
    observed_flux
        Observed flux values.
    model_flux
        Model flux values evaluated at observed wavelengths.
    uncertainty
        One-standard-deviation flux uncertainties.
    include_normalization
        Include the Gaussian normalization term when True.
    validate
        Validate all input arrays when True.

    Returns
    -------
    float
        Gaussian log likelihood.

    Notes
    -----
    With parameter-independent uncertainties, omitting the normalization
    changes the likelihood only by an additive constant.
    """

    observed = np.asarray(
        observed_flux,
        dtype=np.float64,
    )

    model = np.asarray(
        model_flux,
        dtype=np.float64,
    )

    error = np.asarray(
        uncertainty,
        dtype=np.float64,
    )

    if validate:
        _validate_statistical_arrays(
            observed_flux=observed,
            model_flux=model,
            uncertainty=error,
        )

    normalized = (
        observed - model
    ) / error

    chi2_value = np.dot(
        normalized,
        normalized,
    )

    if not include_normalization:
        return float(
            -0.5 * chi2_value
        )

    normalization = np.sum(
        LOG_TWO_PI
        + 2.0 * np.log(error)
    )

    return float(
        -0.5
        * (
            chi2_value
            + normalization
        )
    )


def reduced_chi_square(
    chi2: float,
    n_pixels: int,
    n_free_parameters: int,
) -> float:
    """
    Calculate reduced chi-square.

    Parameters
    ----------
    chi2
        Total chi-square.
    n_pixels
        Number of fitted spectral pixels.
    n_free_parameters
        Number of fitted model parameters.
    """

    n_pixels = int(n_pixels)
    n_free_parameters = int(
        n_free_parameters
    )

    if n_pixels < 1:
        raise ValueError(
            "n_pixels must be at least 1."
        )

    if n_free_parameters < 0:
        raise ValueError(
            "n_free_parameters cannot be negative."
        )

    degrees_of_freedom = (
        n_pixels - n_free_parameters
    )

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Degrees of freedom must be positive. "
            f"Received n_pixels={n_pixels} and "
            f"n_free_parameters={n_free_parameters}."
        )

    return float(
        chi2 / degrees_of_freedom
    )


def _validate_flux_arrays(
    observed_flux: FloatArray,
    model_flux: FloatArray,
) -> None:
    """Validate observed and model flux arrays."""

    if observed_flux.ndim != 1:
        raise ValueError(
            "observed_flux must be one-dimensional."
        )

    if model_flux.ndim != 1:
        raise ValueError(
            "model_flux must be one-dimensional."
        )

    if observed_flux.shape != model_flux.shape:
        raise ValueError(
            "observed_flux and model_flux must have "
            "identical shapes."
        )

    if observed_flux.size == 0:
        raise ValueError(
            "Flux arrays cannot be empty."
        )

    if not np.all(np.isfinite(observed_flux)):
        raise ValueError(
            "observed_flux contains non-finite values."
        )

    if not np.all(np.isfinite(model_flux)):
        raise ValueError(
            "model_flux contains non-finite values."
        )


def _validate_statistical_arrays(
    observed_flux: FloatArray,
    model_flux: FloatArray,
    uncertainty: NDArray[np.float64],
) -> None:
    """Validate flux and uncertainty arrays."""

    _validate_flux_arrays(
        observed_flux=observed_flux,
        model_flux=model_flux,
    )

    if uncertainty.ndim > 1:
        raise ValueError(
            "uncertainty must be a scalar or "
            "one-dimensional array."
        )

    if uncertainty.ndim == 1:
        if uncertainty.shape != observed_flux.shape:
            raise ValueError(
                "Array uncertainty must have the same "
                "shape as observed_flux."
            )

    if not np.all(np.isfinite(uncertainty)):
        raise ValueError(
            "uncertainty contains non-finite values."
        )

    if np.any(uncertainty <= 0.0):
        raise ValueError(
            "All uncertainty values must be positive."
        )
