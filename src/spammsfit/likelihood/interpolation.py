"""Interpolation utilities for SPAMMS model spectra."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def interpolate_model(
    model_wavelength: ArrayLike,
    model_flux: ArrayLike,
    observed_wavelength: ArrayLike,
    *,
    extrapolate: bool = True,
    validate: bool = False,
) -> FloatArray:
    """
    Interpolate a model spectrum onto observed wavelengths.

    Parameters
    ----------
    model_wavelength
        Wavelength array of the SPAMMS model.
    model_flux
        Flux array of the SPAMMS model.
    observed_wavelength
        Wavelength values at which model flux is required.
    extrapolate
        Linearly extrapolate beyond the model boundaries when True.
        When False, raise an error if the observed wavelengths extend
        beyond the model.
    validate
        Perform complete input-array validation. This may be useful for
        direct public use, but should remain False inside repeated
        likelihood evaluations because Spectrum and the SPAMMS output
        reader already validate their arrays.

    Returns
    -------
    numpy.ndarray
        Model flux evaluated at the observed wavelengths.

    Notes
    -----
    NumPy interpolation is used rather than constructing a new
    scipy.interpolate.interp1d object for every likelihood evaluation.
    Linear endpoint extrapolation is applied explicitly when requested.
    """

    model_wavelength_array = np.asarray(
        model_wavelength,
        dtype=np.float64,
    )

    model_flux_array = np.asarray(
        model_flux,
        dtype=np.float64,
    )

    observed_wavelength_array = np.asarray(
        observed_wavelength,
        dtype=np.float64,
    )

    if validate:
        _validate_interpolation_arrays(
            model_wavelength=model_wavelength_array,
            model_flux=model_flux_array,
            observed_wavelength=observed_wavelength_array,
        )

    if model_wavelength_array.size < 2:
        raise ValueError(
            "At least two model wavelength points are "
            "required for linear interpolation."
        )

    model_minimum = model_wavelength_array[0]
    model_maximum = model_wavelength_array[-1]

    observed_below = (
        observed_wavelength_array < model_minimum
    )

    observed_above = (
        observed_wavelength_array > model_maximum
    )

    outside_model = (
        np.any(observed_below)
        or np.any(observed_above)
    )

    if outside_model and not extrapolate:
        observed_minimum = np.min(
            observed_wavelength_array
        )

        observed_maximum = np.max(
            observed_wavelength_array
        )

        raise ValueError(
            "Observed wavelengths extend beyond the "
            "SPAMMS model range. "
            f"Observed range: "
            f"{observed_minimum:.6f}–"
            f"{observed_maximum:.6f}; "
            f"model range: "
            f"{model_minimum:.6f}–"
            f"{model_maximum:.6f}."
        )

    interpolated_flux = np.interp(
        observed_wavelength_array,
        model_wavelength_array,
        model_flux_array,
    )

    if extrapolate and np.any(observed_below):
        lower_slope = (
            model_flux_array[1]
            - model_flux_array[0]
        ) / (
            model_wavelength_array[1]
            - model_wavelength_array[0]
        )

        interpolated_flux[observed_below] = (
            model_flux_array[0]
            + lower_slope
            * (
                observed_wavelength_array[
                    observed_below
                ]
                - model_wavelength_array[0]
            )
        )

    if extrapolate and np.any(observed_above):
        upper_slope = (
            model_flux_array[-1]
            - model_flux_array[-2]
        ) / (
            model_wavelength_array[-1]
            - model_wavelength_array[-2]
        )

        interpolated_flux[observed_above] = (
            model_flux_array[-1]
            + upper_slope
            * (
                observed_wavelength_array[
                    observed_above
                ]
                - model_wavelength_array[-1]
            )
        )

    return interpolated_flux


def _validate_interpolation_arrays(
    model_wavelength: FloatArray,
    model_flux: FloatArray,
    observed_wavelength: FloatArray,
) -> None:
    """Validate arrays supplied directly to interpolate_model()."""

    if model_wavelength.ndim != 1:
        raise ValueError(
            "model_wavelength must be one-dimensional."
        )

    if model_flux.ndim != 1:
        raise ValueError(
            "model_flux must be one-dimensional."
        )

    if observed_wavelength.ndim != 1:
        raise ValueError(
            "observed_wavelength must be one-dimensional."
        )

    if model_wavelength.size < 2:
        raise ValueError(
            "model_wavelength must contain at least "
            "two values."
        )

    if observed_wavelength.size == 0:
        raise ValueError(
            "observed_wavelength cannot be empty."
        )

    if model_wavelength.shape != model_flux.shape:
        raise ValueError(
            "model_wavelength and model_flux must have "
            "identical shapes."
        )

    if not np.all(np.isfinite(model_wavelength)):
        raise ValueError(
            "model_wavelength contains non-finite values."
        )

    if not np.all(np.isfinite(model_flux)):
        raise ValueError(
            "model_flux contains non-finite values."
        )

    if not np.all(np.isfinite(observed_wavelength)):
        raise ValueError(
            "observed_wavelength contains non-finite values."
        )

    if not np.all(
        np.diff(model_wavelength) > 0.0
    ):
        raise ValueError(
            "model_wavelength must be strictly increasing."
        )

    if not np.all(
        np.diff(observed_wavelength) > 0.0
    ):
        raise ValueError(
            "observed_wavelength must be strictly increasing."
        )
