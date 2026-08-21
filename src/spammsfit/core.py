"""Core spectral-fitting interface for SPAMMSFit."""
from __future__ import annotations
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict
import numpy as np
from numpy.typing import ArrayLike, NDArray
from spammsfit.configuration import SpammsConfig
from spammsfit.forward.input import InputBuilder
from spammsfit.forward.output import ModelSpectra
from spammsfit.forward.runner import SpammsRunner
from spammsfit.likelihood.interpolation import (interpolate_model)
from spammsfit.likelihood.statistics import (chi_square, reduced_chi_square)
from spammsfit.parameters import ParameterSet
from spammsfit.spectrum import LineData, Spectrum

if TYPE_CHECKING:
    from spammsfit.results.model import ModelResult

FloatArray = NDArray[np.float64]
class EvaluationDetails(TypedDict):
    """
    Detailed output from one explicitly requested model evaluation.
    """

    parameters: dict[str, float | int]

    observed_wavelengths: dict[
        str,
        FloatArray,
    ]

    observed_fluxes: dict[
        str,
        FloatArray,
    ]

    observed_uncertainties: dict[
        str,
        FloatArray,
    ]

    models: ModelSpectra

    interpolated_models: dict[
        str,
        FloatArray,
    ]

    residuals: dict[
        str,
        FloatArray,
    ]

    chi2_by_line: dict[
        str,
        float,
    ]

    chi2: float
    reduced_chi2: float
    log_likelihood: float

class SpammsFit:
    """
    Common SPAMMS spectral-fitting calculation.

    Parameters
    ----------
    spectrum
        Observed spectrum containing all selected fitting lines.
    parameters
        Fixed/free parameter configuration.
    config
        SPAMMS paths and execution settings.
    extrapolate
        Allow linear extrapolation when an observed line extends beyond
        the corresponding SPAMMS model wavelength range.

    Notes
    -----
    SpammsFit performs the shared scientific calculation. Parameter-space
    exploration is handled separately by ChiSquareFit, DEFit and
    BayesianFit.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        parameters: ParameterSet,
        config: SpammsConfig,
        *,
        extrapolate: bool = True,
    ) -> None:
        if not isinstance(spectrum, Spectrum):
            raise TypeError(
                "spectrum must be a Spectrum instance."
            )

        if not isinstance(parameters, ParameterSet):
            raise TypeError(
                "parameters must be a ParameterSet instance."
            )

        if not isinstance(config, SpammsConfig):
            raise TypeError(
                "config must be a SpammsConfig instance."
            )

        if spectrum.n_lines == 0:
            raise ValueError(
                "The observed Spectrum has no selected lines. "
                "Use spectrum.add_line() before constructing "
                "SpammsFit."
            )

        self.spectrum = spectrum
        self.parameters = parameters
        self.config = config
        self.extrapolate = bool(extrapolate)

        # The observed line selection is authoritative. SPAMMS must
        # generate exactly the same named lines.
        self.parameters.set_lines(
            self.spectrum.line_names,
        )

        self.input_builder = InputBuilder(
            template_file=self.parameters.input_file,
        )

        self.runner = SpammsRunner(
            config=self.config,
            input_builder=self.input_builder,
        )

        # Prepare references to the observed line arrays once. These
        # arrays do not need to be selected again during every model
        # evaluation.
        self._observed_lines = {
            line_name: self.spectrum.get_line(
                line_name
            )
            for line_name in self.spectrum.line_names
        }

        # The Gaussian normalization depends only on the fixed observed
        # uncertainties, so calculate it once rather than during every
        # Bayesian likelihood evaluation.
        self._log_normalization_by_line = {
            line_name: self._calculate_log_normalization(
                line_data
            )
            for line_name, line_data
            in self._observed_lines.items()
        }

        self._n_model_evaluations = 0
        self._n_chi2_evaluations = 0
        self._n_log_likelihood_evaluations = 0

    def parameter_values(
        self,
        theta: ArrayLike,
        *,
        check_bounds: bool = False,
    ) -> dict[str, float | int]:
        """
        Expand a free-parameter vector into complete model values.
        """

        return self.parameters.vector_to_values(
            theta,
            check_bounds=check_bounds,
        )

    def run_model(
        self,
        theta: ArrayLike,
        *,
        check_bounds: bool = False,
    ) -> ModelSpectra:
        """
        Run one multiline SPAMMS model for a free-parameter vector.

        SPAMMS is executed exactly once, irrespective of the number of
        selected spectral lines.
        """

        parameter_values = self.parameter_values(
            theta,
            check_bounds=check_bounds,
        )

        return self.run_model_from_values(
            parameter_values,
        )

    def run_model_from_values(
        self,
        parameter_values: Mapping[str, float | int],
    ) -> ModelSpectra:
        """
        Run one SPAMMS model from complete named parameter values.

        This method is useful when values come from a grid rather than
        a continuous optimizer vector.
        """

        models = self.runner.run(
            parameter_values=parameter_values,
            selected_lines=self.spectrum.line_names,
        )

        self._validate_model_lines(models)
        self._n_model_evaluations += 1

        return models

    def chi2(
        self,
        theta: ArrayLike,
    ) -> float:
        """
        Calculate total multiline chi-square for one parameter vector.

        Only the scalar chi-square is retained. Model arrays are released
        after this method returns.
        """

        models = self.run_model(theta)

        total_chi2 = self._calculate_total_chi2(
            models,
        )

        self._n_chi2_evaluations += 1

        return total_chi2

    def log_likelihood(
        self,
        theta: ArrayLike,
        *,
        include_normalization: bool = True,
    ) -> float:
        """
        Calculate the total multiline Gaussian log likelihood.

        Only the scalar likelihood is retained. Model arrays are released
        after this method returns.
        """

        models = self.run_model(theta)

        total_chi2 = 0.0

        for line_name, observed in (
            self._observed_lines.items()
        ):
            model_wave, model_flux = models[
                line_name
            ]

            model_interpolated = interpolate_model(
                model_wavelength=model_wave,
                model_flux=model_flux,
                observed_wavelength=observed.wavelength,
                extrapolate=self.extrapolate,
            )

            total_chi2 += chi_square(
                observed_flux=observed.flux,
                model_flux=model_interpolated,
                uncertainty=observed.uncertainty,
            )

        if include_normalization:
            normalization = sum(
                self._log_normalization_by_line.values()
            )
        else:
            normalization = 0.0

        self._n_log_likelihood_evaluations += 1

        return float(
            -0.5
            * (
                total_chi2
                + normalization
            )
        )

    def evaluate(
        self,
        theta: ArrayLike,
    ) -> EvaluationDetails:
        """
        Perform one detailed multiline model evaluation.

        This method retains the observed arrays, interpolated models and
        residuals in its returned dictionary. It should be used for final
        solutions and diagnostics, not for every Bayesian likelihood
        evaluation.
        """

        parameter_values = self.parameter_values(
            theta,
        )

        return self.evaluate_from_values(
            parameter_values,
            n_free_parameters=self.parameters.n_free,
        )

    def evaluate_from_values(
        self,
        parameter_values: Mapping[str, float | int],
        *,
        n_free_parameters: int = 0,
    ) -> EvaluationDetails:
        """
        Perform one detailed evaluation from complete named values.

        Parameters
        ----------
        parameter_values
            Complete numerical mapping for one SPAMMS model.
        n_free_parameters
            Number of parameters estimated from the data when calculating
            reduced chi-square. Use zero for a manually selected preview.
        """
        if isinstance(n_free_parameters, bool) or int(n_free_parameters) != n_free_parameters:
            raise TypeError("n_free_parameters must be an integer.")

        n_free_parameters = int(n_free_parameters)
        if n_free_parameters < 0:
            raise ValueError("n_free_parameters cannot be negative.")

        parameter_values = dict(parameter_values)
        models = self.run_model_from_values(parameter_values)

        observed_wavelengths: dict[
            str,
            FloatArray,
        ] = {}

        observed_fluxes: dict[
            str,
            FloatArray,
        ] = {}

        observed_uncertainties: dict[
            str,
            FloatArray,
        ] = {}

        interpolated_models: dict[
            str,
            FloatArray,
        ] = {}

        residual_arrays: dict[
            str,
            FloatArray,
        ] = {}

        chi2_by_line: dict[
            str,
            float,
        ] = {}

        for line_name, observed in (
            self._observed_lines.items()
        ):
            model_wave, model_flux = models[
                line_name
            ]

            model_interpolated = interpolate_model(
                model_wavelength=model_wave,
                model_flux=model_flux,
                observed_wavelength=(
                    observed.wavelength
                ),
                extrapolate=self.extrapolate,
            )

            line_residual = (
                observed.flux
                - model_interpolated
            )

            line_chi2 = chi_square(
                observed_flux=observed.flux,
                model_flux=model_interpolated,
                uncertainty=observed.uncertainty,
            )

            # Retain references to the observed arrays for final plotting
            # and diagnostic output.
            observed_wavelengths[line_name] = (
                observed.wavelength
            )

            observed_fluxes[line_name] = (
                observed.flux
            )

            observed_uncertainties[line_name] = (
                observed.uncertainty
            )

            interpolated_models[line_name] = (
                model_interpolated
            )

            residual_arrays[line_name] = (
                line_residual
            )

            chi2_by_line[line_name] = (
                line_chi2
            )

        total_chi2 = float(
            sum(chi2_by_line.values())
        )

        total_normalization = float(
            sum(
                self._log_normalization_by_line.values()
            )
        )

        log_likelihood = float(
            -0.5
            * (
                total_chi2
                + total_normalization
            )
        )

        reduced = reduced_chi_square(
            chi2=total_chi2,
            n_pixels=self.n_fitting_pixels,
            n_free_parameters=n_free_parameters,
        )

        return {
            "parameters": parameter_values,
            "observed_wavelengths": (
                observed_wavelengths
            ),
            "observed_fluxes": (
                observed_fluxes
            ),
            "observed_uncertainties": (
                observed_uncertainties
            ),
            "models": models,
            "interpolated_models": (
                interpolated_models
            ),
            "residuals": residual_arrays,
            "chi2_by_line": chi2_by_line,
            "chi2": total_chi2,
            "reduced_chi2": reduced,
            "log_likelihood": log_likelihood,
        }

    def preview_model(
        self,
        **parameter_values: float | int,
    ) -> ModelResult:
        """
        Generate and evaluate one user-selected SPAMMS model.

        Supplied values temporarily override the current ParameterSet
        values. Fitting bounds and fixed/free states are ignored, and the
        ParameterSet and original SPAMMS input file remain unchanged.
        All lines selected in Spectrum are calculated in one SPAMMS run.
        """
        from spammsfit.results.model import ModelResult

        preview_values = self.parameters.with_values(**parameter_values)
        start_time = time.perf_counter()
        evaluation = self.evaluate_from_values(
            preview_values,
            n_free_parameters=0,
        )
        runtime = time.perf_counter() - start_time

        return ModelResult(
            evaluation=evaluation,
            runtime=runtime,
            metadata={
                "selected_lines": list(self.spectrum.line_names),
                "parameter_overrides": dict(parameter_values),
            },
        )

    def _calculate_total_chi2(
        self,
        models: ModelSpectra,
    ) -> float:
        """Calculate total chi-square from prepared model spectra."""

        total_chi2 = 0.0

        for line_name, observed in (
            self._observed_lines.items()
        ):
            model_wave, model_flux = models[
                line_name
            ]

            model_interpolated = interpolate_model(
                model_wavelength=model_wave,
                model_flux=model_flux,
                observed_wavelength=observed.wavelength,
                extrapolate=self.extrapolate,
            )

            total_chi2 += chi_square(
                observed_flux=observed.flux,
                model_flux=model_interpolated,
                uncertainty=observed.uncertainty,
            )

        return float(total_chi2)

    def _validate_model_lines(
        self,
        models: ModelSpectra,
    ) -> None:
        """Ensure that model and observed line names agree exactly."""

        expected = set(
            self.spectrum.line_names
        )

        received = set(models)

        missing = expected - received
        unexpected = received - expected

        if missing or unexpected:
            raise ValueError(
                "SPAMMS model lines do not match the "
                "observed line selection. "
                f"Missing: {sorted(missing)}; "
                f"unexpected: {sorted(unexpected)}."
            )

    @staticmethod
    def _calculate_log_normalization(
        observed: LineData,
    ) -> float:
        """
        Calculate the fixed Gaussian normalization for one line.

        This returns:

        sum(log(2*pi*uncertainty**2)).
        """

        return float(
            np.sum(
                np.log(2.0 * np.pi)
                + 2.0
                * np.log(
                    observed.uncertainty
                )
            )
        )

    def timing_summary(self) -> str:
        """Return SPAMMS execution timing information."""

        return self.runner.timing_summary()

    def summary(self) -> str:
        """Return a readable summary of this fitting setup."""

        return (
            "SPAMMSFit configuration\n"
            "=======================\n"
            f"Observed spectrum: "
            f"{self.spectrum.name or 'unnamed'}\n"
            f"Selected lines: "
            f"{', '.join(self.spectrum.line_names)}\n"
            f"Fitting pixels: "
            f"{self.n_fitting_pixels}\n"
            f"Free parameters: "
            f"{', '.join(self.parameters.free_names()) or 'none'}\n"
            f"Model evaluations: "
            f"{self._n_model_evaluations}"
        )

    @property
    def n_fitting_pixels(self) -> int:
        """Return the number of observed pixels included in the fit."""

        return self.spectrum.n_fitting_pixels

    @property
    def n_model_evaluations(self) -> int:
        """Return the number of completed SPAMMS calculations."""

        return self._n_model_evaluations

    @property
    def n_chi2_evaluations(self) -> int:
        """Return the number of scalar chi-square evaluations."""

        return self._n_chi2_evaluations

    @property
    def n_log_likelihood_evaluations(self) -> int:
        """Return the number of scalar likelihood evaluations."""

        return self._n_log_likelihood_evaluations

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"SpammsFit("
            f"n_lines={self.spectrum.n_lines}, "
            f"n_free={self.parameters.n_free}, "
            f"n_model_evaluations="
            f"{self._n_model_evaluations}"
            f")"
        )
