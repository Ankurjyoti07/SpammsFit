"""Shared result container for SPAMMSFit fitting methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spammsfit.core import EvaluationDetails
from spammsfit.forward.output import ModelSpectra


class BaseResult:
    """
    Store outputs common to all SPAMMSFit fitting methods.

    Parameters
    ----------
    method
        Human-readable fitting-method name.
    best_parameters
        Complete fixed and fitted parameter values for the best model.
    free_parameter_names
        Names of parameters explored by the fitting method.
    chi2
        Total chi-square of the best model.
    reduced_chi2
        Reduced chi-square of the best model.
    log_likelihood
        Gaussian log likelihood of the best model.
    runtime
        Total fitting runtime in seconds.
    n_evaluations
        Number of model evaluations performed by the fitting method.
    evaluation
        Detailed final evaluation containing the best model,
        interpolated model arrays and residuals.
    metadata
        Optional additional result metadata.
    """

    def __init__(
        self,
        *,
        method: str,
        best_parameters: Mapping[str, float | int],
        free_parameter_names: tuple[str, ...],
        chi2: float,
        reduced_chi2: float,
        log_likelihood: float,
        runtime: float,
        n_evaluations: int,
        evaluation: EvaluationDetails,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.method = str(method)

        self.best_parameters = dict(
            best_parameters
        )

        self.free_parameter_names = tuple(
            free_parameter_names
        )

        self.chi2 = float(chi2)
        self.reduced_chi2 = float(
            reduced_chi2
        )

        self.log_likelihood = float(
            log_likelihood
        )

        self.runtime = float(runtime)
        self.n_evaluations = int(
            n_evaluations
        )

        self.evaluation = evaluation

        self.metadata = (
            {}
            if metadata is None
            else dict(metadata)
        )

        self._validate()

    def _validate(self) -> None:
        """Validate common result values."""

        if not self.method.strip():
            raise ValueError(
                "Result method name cannot be empty."
            )

        if not self.best_parameters:
            raise ValueError(
                "best_parameters cannot be empty."
            )

        if not np.isfinite(self.chi2):
            raise ValueError(
                "chi2 must be finite."
            )

        if self.chi2 < 0.0:
            raise ValueError(
                "chi2 cannot be negative."
            )

        if not np.isfinite(self.reduced_chi2):
            raise ValueError(
                "reduced_chi2 must be finite."
            )

        if self.reduced_chi2 < 0.0:
            raise ValueError(
                "reduced_chi2 cannot be negative."
            )

        if not np.isfinite(self.log_likelihood):
            raise ValueError(
                "log_likelihood must be finite."
            )

        if not np.isfinite(self.runtime):
            raise ValueError(
                "runtime must be finite."
            )

        if self.runtime < 0.0:
            raise ValueError(
                "runtime cannot be negative."
            )

        if self.n_evaluations < 1:
            raise ValueError(
                "n_evaluations must be at least 1."
            )

        evaluation_parameters = (
            self.evaluation["parameters"]
        )

        if self.best_parameters != (
            evaluation_parameters
        ):
            raise ValueError(
                "best_parameters do not match the "
                "parameters stored in evaluation."
            )

        if not np.isclose(
            self.chi2,
            self.evaluation["chi2"],
        ):
            raise ValueError(
                "chi2 does not match the detailed "
                "evaluation."
            )

        if not np.isclose(
            self.reduced_chi2,
            self.evaluation["reduced_chi2"],
        ):
            raise ValueError(
                "reduced_chi2 does not match the "
                "detailed evaluation."
            )

    def get_parameter(
        self,
        name: str,
    ) -> float | int:
        """Return one best-fitting parameter value."""

        try:
            return self.best_parameters[name]
        except KeyError as error:
            available = ", ".join(
                self.best_parameters
            )

            raise KeyError(
                f"Unknown result parameter {name!r}. "
                f"Available parameters: {available}."
            ) from error

    def best_model(
        self,
        line_name: str,
        *,
        interpolated: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the best model for one spectral line.

        Parameters
        ----------
        line_name
            Name of the requested spectral line.
        interpolated
            When True, return the model evaluated at the observed
            wavelengths. When False, return the model on its native
            SPAMMS wavelength grid.

        Returns
        -------
        wavelength, flux
            Wavelength and best-model flux arrays.
        """

        if line_name not in self.line_names:
            available = ", ".join(
                self.line_names
            )

            raise KeyError(
                f"No best model is available for "
                f"line {line_name!r}. "
                f"Available lines: {available}."
            )

        if interpolated:
            try:
                wavelength = self.evaluation[
                    "observed_wavelengths"
                ][line_name]

                flux = self.evaluation[
                    "interpolated_models"
                ][line_name]

            except KeyError as error:
                raise KeyError(
                    f"No interpolated best model is "
                    f"available for line {line_name!r}."
                ) from error

        else:
            try:
                wavelength, flux = self.evaluation[
                    "models"
                ][line_name]

            except KeyError as error:
                raise KeyError(
                    f"No native SPAMMS model is "
                    f"available for line {line_name!r}."
                ) from error

        if wavelength.ndim != 1:
            raise ValueError(
                f"Best-model wavelength array for "
                f"{line_name!r} is not one-dimensional."
            )

        if flux.ndim != 1:
            raise ValueError(
                f"Best-model flux array for "
                f"{line_name!r} is not one-dimensional."
            )

        if wavelength.shape != flux.shape:
            raise ValueError(
                f"Best-model wavelength and flux arrays "
                f"for {line_name!r} have different shapes."
            )

        return wavelength, flux

    def residuals(
        self,
        line_name: str,
    ) -> np.ndarray:
        """Return observed-minus-model residuals for one line."""

        try:
            return self.evaluation[
                "residuals"
            ][line_name]
        except KeyError as error:
            raise KeyError(
                f"No residuals are available for "
                f"line {line_name!r}."
            ) from error

    def chi2_for_line(
        self,
        line_name: str,
    ) -> float:
        """Return the chi-square contribution from one line."""

        try:
            return self.evaluation[
                "chi2_by_line"
            ][line_name]
        except KeyError as error:
            raise KeyError(
                f"No chi-square value is available for "
                f"line {line_name!r}."
            ) from error

    @property
    def line_names(self) -> tuple[str, ...]:
        """Return line names included in the result."""

        return tuple(
            self.evaluation[
                "chi2_by_line"
            ]
        )

    @property
    def models(self) -> ModelSpectra:
        """Return native best-fitting SPAMMS line profiles."""

        return self.evaluation["models"]

    @property
    def interpolated_models(
        self,
    ) -> dict[str, np.ndarray]:
        """Return best models interpolated to observed wavelengths."""

        return self.evaluation[
            "interpolated_models"
        ]

    @property
    def residual_arrays(
        self,
    ) -> dict[str, np.ndarray]:
        """Return residual arrays for every fitted line."""

        return self.evaluation["residuals"]

    @property
    def chi2_by_line(
        self,
    ) -> dict[str, float]:
        """Return chi-square contributions for all lines."""

        return self.evaluation["chi2_by_line"]

    def to_dict(self) -> dict[str, Any]:
        """
        Return scalar result information as a serializable dictionary.

        Large numerical model and residual arrays are intentionally
        excluded.
        """

        return {
            "method": self.method,
            "best_parameters": dict(
                self.best_parameters
            ),
            "free_parameter_names": list(
                self.free_parameter_names
            ),
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "log_likelihood": (
                self.log_likelihood
            ),
            "runtime_seconds": self.runtime,
            "n_evaluations": (
                self.n_evaluations
            ),
            "chi2_by_line": dict(
                self.chi2_by_line
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    def save_summary(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save scalar result information as JSON.

        Model arrays and method-specific outputs will be saved separately
        by the method-specific result classes.
        """

        output_path = (
            Path(path)
            .expanduser()
            .resolve()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return output_path

    def summary(self) -> str:
        """Return a readable result summary."""

        lines = [
            f"Method: {self.method}",
            f"Chi-square: {self.chi2:.6f}",
            (
                "Reduced chi-square: "
                f"{self.reduced_chi2:.6f}"
            ),
            (
                "Log likelihood: "
                f"{self.log_likelihood:.6f}"
            ),
            (
                "Model evaluations: "
                f"{self.n_evaluations}"
            ),
            (
                "Runtime: "
                f"{self.runtime:.3f} seconds"
            ),
            "Best parameters:",
        ]

        for name, value in (
            self.best_parameters.items()
        ):
            free_marker = (
                "free"
                if name
                in self.free_parameter_names
                else "fixed"
            )

            lines.append(
                f"  {name}: {value} "
                f"({free_marker})"
            )

        lines.append(
            "Chi-square by line:"
        )

        for line_name, value in (
            self.chi2_by_line.items()
        ):
            lines.append(
                f"  {line_name}: {value:.6f}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"{self.__class__.__name__}("
            f"method={self.method!r}, "
            f"chi2={self.chi2:.6f}, "
            f"n_evaluations={self.n_evaluations}"
            f")"
        )
