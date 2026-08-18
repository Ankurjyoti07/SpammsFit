"""Result container for Bayesian SPAMMSFit inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from spammsfit.core import EvaluationDetails
from spammsfit.results.base import BaseResult


FloatArray = NDArray[np.float64]


class BayesianResult(BaseResult):
    """
    Store an emcee Bayesian inference result.

    Parameters
    ----------
    chain
        MCMC chain with shape ``(nsteps, nwalkers, ndim)``.
    log_probability
        Log-probability chain with shape ``(nsteps, nwalkers)``.
    posterior_samples
        Flattened samples after burn-in removal and thinning.
    acceptance_fraction
        Acceptance fraction for each walker.
    credible_intervals
        Table containing the 16th, 50th and 84th percentiles.
    posterior_median_vector
        Median free-parameter vector.
    maximum_probability_vector
        Sample with the highest stored log probability.
    autocorrelation_time
        Estimated autocorrelation time for each parameter, when
        available.
    best_parameters
        Complete parameter mapping evaluated at the posterior median.
    free_parameter_names
        Names corresponding to the chain's final dimension.
    settings
        Sampler settings such as walkers, steps, burn-in and thinning.
    runtime
        Total fitting runtime in seconds.
    n_evaluations
        Number of SPAMMS likelihood evaluations.
    evaluation
        Detailed SPAMMS evaluation at the posterior median.
    backend_file
        Optional HDF5 backend used during sampling.
    metadata
        Optional additional metadata.
    """

    def __init__(
        self,
        *,
        chain: ArrayLike,
        log_probability: ArrayLike,
        posterior_samples: ArrayLike,
        acceptance_fraction: ArrayLike,
        credible_intervals: pd.DataFrame,
        posterior_median_vector: ArrayLike,
        maximum_probability_vector: ArrayLike,
        maximum_log_probability: float,
        autocorrelation_time: ArrayLike | None,
        best_parameters: Mapping[str, float | int],
        free_parameter_names: tuple[str, ...],
        settings: Mapping[str, Any],
        runtime: float,
        n_evaluations: int,
        evaluation: EvaluationDetails,
        backend_file: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.chain = np.asarray(
            chain,
            dtype=np.float64,
        ).copy()

        self.log_probability = np.asarray(
            log_probability,
            dtype=np.float64,
        ).copy()

        self.posterior_samples = np.asarray(
            posterior_samples,
            dtype=np.float64,
        ).copy()

        self.acceptance_fraction = np.asarray(
            acceptance_fraction,
            dtype=np.float64,
        ).copy()

        self.credible_intervals = (
            self._prepare_credible_intervals(
                credible_intervals
            )
        )

        self.posterior_median_vector = np.asarray(
            posterior_median_vector,
            dtype=np.float64,
        ).copy()

        self.maximum_probability_vector = np.asarray(
            maximum_probability_vector,
            dtype=np.float64,
        ).copy()

        self.maximum_log_probability = float(
            maximum_log_probability
        )

        self.autocorrelation_time = (
            None
            if autocorrelation_time is None
            else np.asarray(
                autocorrelation_time,
                dtype=np.float64,
            ).copy()
        )

        self.settings = dict(settings)

        self.backend_file = (
            None
            if backend_file is None
            else (
                Path(backend_file)
                .expanduser()
                .resolve()
            )
        )

        self._validate_bayesian_arrays(
            free_parameter_names=(
                free_parameter_names
            )
        )

        combined_metadata = {
            "sampler": "emcee.EnsembleSampler",
            "backend_file": (
                None
                if self.backend_file is None
                else str(self.backend_file)
            ),
        }

        if metadata is not None:
            combined_metadata.update(
                dict(metadata)
            )

        super().__init__(
            method="Bayesian inference",
            best_parameters=best_parameters,
            free_parameter_names=(
                free_parameter_names
            ),
            chi2=evaluation["chi2"],
            reduced_chi2=(
                evaluation["reduced_chi2"]
            ),
            log_likelihood=(
                evaluation["log_likelihood"]
            ),
            runtime=runtime,
            n_evaluations=n_evaluations,
            evaluation=evaluation,
            metadata=combined_metadata,
        )

    @staticmethod
    def _prepare_credible_intervals(
        intervals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and copy the credible-interval table."""

        if not isinstance(
            intervals,
            pd.DataFrame,
        ):
            raise TypeError(
                "credible_intervals must be "
                "a pandas DataFrame."
            )

        required_columns = {
            "parameter",
            "p16",
            "p50",
            "p84",
            "minus",
            "plus",
        }

        missing = (
            required_columns
            - set(intervals.columns)
        )

        if missing:
            raise ValueError(
                "credible_intervals is missing required "
                f"columns: {sorted(missing)}."
            )

        if intervals.empty:
            raise ValueError(
                "credible_intervals cannot be empty."
            )

        prepared = intervals.copy().reset_index(
            drop=True
        )

        numerical_columns = [
            "p16",
            "p50",
            "p84",
            "minus",
            "plus",
        ]

        values = prepared[
            numerical_columns
        ].to_numpy(
            dtype=float
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "credible_intervals contains "
                "non-finite values."
            )

        if np.any(
            prepared["p16"]
            > prepared["p50"]
        ):
            raise ValueError(
                "Some p16 values exceed p50."
            )

        if np.any(
            prepared["p50"]
            > prepared["p84"]
        ):
            raise ValueError(
                "Some p50 values exceed p84."
            )

        return prepared

    def _validate_bayesian_arrays(
        self,
        free_parameter_names: tuple[str, ...],
    ) -> None:
        """Validate chain and posterior array dimensions."""

        n_parameters = len(
            free_parameter_names
        )

        if n_parameters < 1:
            raise ValueError(
                "BayesianResult requires at least "
                "one free parameter."
            )

        if self.chain.ndim != 3:
            raise ValueError(
                "chain must have shape "
                "(nsteps, nwalkers, ndim)."
            )

        n_steps, n_walkers, chain_parameters = (
            self.chain.shape
        )

        if n_steps < 1:
            raise ValueError(
                "The MCMC chain contains no steps."
            )

        if n_walkers < 1:
            raise ValueError(
                "The MCMC chain contains no walkers."
            )

        if chain_parameters != n_parameters:
            raise ValueError(
                "The chain dimension does not match "
                "the number of free parameters."
            )

        if self.log_probability.shape != (
            n_steps,
            n_walkers,
        ):
            raise ValueError(
                "log_probability must have shape "
                "(nsteps, nwalkers)."
            )

        if self.posterior_samples.ndim != 2:
            raise ValueError(
                "posterior_samples must be "
                "two-dimensional."
            )

        if (
            self.posterior_samples.shape[1]
            != n_parameters
        ):
            raise ValueError(
                "posterior_samples width does not match "
                "the number of free parameters."
            )

        if self.posterior_samples.shape[0] < 1:
            raise ValueError(
                "No posterior samples remain after "
                "burn-in removal and thinning."
            )

        if self.acceptance_fraction.shape != (
            n_walkers,
        ):
            raise ValueError(
                "acceptance_fraction must contain "
                "one value per walker."
            )

        if self.posterior_median_vector.shape != (
            n_parameters,
        ):
            raise ValueError(
                "posterior_median_vector has an "
                "incorrect shape."
            )

        if self.maximum_probability_vector.shape != (
            n_parameters,
        ):
            raise ValueError(
                "maximum_probability_vector has an "
                "incorrect shape."
            )

        if (
            len(self.credible_intervals)
            != n_parameters
        ):
            raise ValueError(
                "credible_intervals must contain one "
                "row per free parameter."
            )

        interval_names = tuple(
            self.credible_intervals[
                "parameter"
            ]
        )

        if interval_names != free_parameter_names:
            raise ValueError(
                "credible_intervals parameter order "
                "does not match free_parameter_names."
            )

        arrays_to_check = [
            self.chain,
            self.posterior_samples,
            self.acceptance_fraction,
            self.posterior_median_vector,
            self.maximum_probability_vector,
        ]

        for array in arrays_to_check:
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "A Bayesian result array contains "
                    "non-finite values."
                )

        if not np.all(
            np.isfinite(
                self.log_probability
            )
            | np.isneginf(
                self.log_probability
            )
        ):
            raise ValueError(
                "log_probability contains invalid values."
            )

        if not np.isfinite(
            self.maximum_log_probability
        ):
            raise ValueError(
                "maximum_log_probability must be finite."
            )

        if np.any(
            self.acceptance_fraction < 0.0
        ) or np.any(
            self.acceptance_fraction > 1.0
        ):
            raise ValueError(
                "Acceptance fractions must lie "
                "between 0 and 1."
            )

        if self.autocorrelation_time is not None:
            if self.autocorrelation_time.shape != (
                n_parameters,
            ):
                raise ValueError(
                    "autocorrelation_time has an "
                    "incorrect shape."
                )

        self.chain.flags.writeable = False
        self.log_probability.flags.writeable = False
        self.posterior_samples.flags.writeable = False
        self.acceptance_fraction.flags.writeable = False
        self.posterior_median_vector.flags.writeable = False
        self.maximum_probability_vector.flags.writeable = False

        if self.autocorrelation_time is not None:
            self.autocorrelation_time.flags.writeable = False

    @property
    def n_steps(self) -> int:
        """Return the number of stored MCMC steps."""

        return int(
            self.chain.shape[0]
        )

    @property
    def n_walkers(self) -> int:
        """Return the number of walkers."""

        return int(
            self.chain.shape[1]
        )

    @property
    def n_parameters(self) -> int:
        """Return the number of sampled parameters."""

        return int(
            self.chain.shape[2]
        )

    @property
    def n_posterior_samples(self) -> int:
        """Return the number of flattened posterior samples."""

        return int(
            self.posterior_samples.shape[0]
        )

    @property
    def mean_acceptance_fraction(self) -> float:
        """Return the mean acceptance fraction."""

        return float(
            np.mean(
                self.acceptance_fraction
            )
        )

    @property
    def median_parameters(
        self,
    ) -> dict[str, float]:
        """Return posterior median values by parameter name."""

        return {
            name: float(value)
            for name, value in zip(
                self.free_parameter_names,
                self.posterior_median_vector,
                strict=True,
            )
        }

    @property
    def maximum_probability_parameters(
        self,
    ) -> dict[str, float]:
        """Return the maximum-probability sampled values."""

        return {
            name: float(value)
            for name, value in zip(
                self.free_parameter_names,
                self.maximum_probability_vector,
                strict=True,
            )
        }

    def interval(
        self,
        parameter: str,
    ) -> pd.Series:
        """Return credible-interval information for one parameter."""

        selected = self.credible_intervals[
            self.credible_intervals[
                "parameter"
            ]
            == parameter
        ]

        if selected.empty:
            raise KeyError(
                f"No posterior interval is available "
                f"for parameter {parameter!r}."
            )

        return selected.iloc[0].copy()

    def to_dict(self) -> dict[str, Any]:
        """Return serializable Bayesian result information."""

        result = super().to_dict()

        result.update(
            {
                "n_steps": self.n_steps,
                "n_walkers": self.n_walkers,
                "n_parameters": (
                    self.n_parameters
                ),
                "n_posterior_samples": (
                    self.n_posterior_samples
                ),
                "mean_acceptance_fraction": (
                    self.mean_acceptance_fraction
                ),
                "maximum_log_probability": (
                    self.maximum_log_probability
                ),
                "posterior_median_vector": (
                    self.posterior_median_vector.tolist()
                ),
                "maximum_probability_vector": (
                    self.maximum_probability_vector.tolist()
                ),
                "credible_intervals": (
                    self.credible_intervals.to_dict(
                        orient="records"
                    )
                ),
                "autocorrelation_time": (
                    None
                    if self.autocorrelation_time is None
                    else (
                        self.autocorrelation_time.tolist()
                    )
                ),
                "settings": dict(
                    self.settings
                ),
                "backend_file": (
                    None
                    if self.backend_file is None
                    else str(
                        self.backend_file
                    )
                ),
            }
        )

        return result

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """
        Save Bayesian summary, posterior arrays and best model.

        The HDF5 backend is not copied because it is written continuously
        during sampling.
        """

        output_directory = (
            Path(directory)
            .expanduser()
            .resolve()
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_directory
            / "summary.json"
        ).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.credible_intervals.to_csv(
            output_directory
            / "credible_intervals.csv",
            index=False,
        )

        np.save(
            output_directory
            / "posterior_samples.npy",
            self.posterior_samples,
        )

        np.savetxt(
            output_directory
            / "posterior_samples.txt",
            self.posterior_samples,
            header=" ".join(
                self.free_parameter_names
            ),
        )

        np.save(
            output_directory
            / "chain.npy",
            self.chain,
        )

        np.save(
            output_directory
            / "log_probability.npy",
            self.log_probability,
        )

        best_model_arrays: dict[
            str,
            np.ndarray,
        ] = {}

        for line_name, (
            model_wavelength,
            model_flux,
        ) in self.models.items():
            best_model_arrays[
                f"{line_name}_model_wavelength"
            ] = model_wavelength

            best_model_arrays[
                f"{line_name}_model_flux"
            ] = model_flux

            best_model_arrays[
                f"{line_name}_interpolated_flux"
            ] = self.interpolated_models[
                line_name
            ]

            best_model_arrays[
                f"{line_name}_residual"
            ] = self.residual_arrays[
                line_name
            ]

        np.savez_compressed(
            output_directory
            / "best_model.npz",
            **best_model_arrays,
        )

        return output_directory

    def summary(self) -> str:
        """Return a readable Bayesian result summary."""

        lines = [
            "Bayesian inference",
            "==================",
            "",
            f"Walkers: {self.n_walkers}",
            f"Steps: {self.n_steps}",
            (
                "Posterior samples: "
                f"{self.n_posterior_samples}"
            ),
            (
                "Mean acceptance fraction: "
                f"{self.mean_acceptance_fraction:.4f}"
            ),
            (
                "Maximum log probability: "
                f"{self.maximum_log_probability:.6f}"
            ),
            (
                "Runtime: "
                f"{self.runtime:.3f} seconds"
            ),
            (
                "SPAMMS evaluations: "
                f"{self.n_evaluations}"
            ),
            "",
            "Posterior credible intervals:",
        ]

        for row in (
            self.credible_intervals.itertuples(
                index=False
            )
        ):
            lines.append(
                f"  {row.parameter}: "
                f"{row.p50:.6g} "
                f"-{row.minus:.6g} "
                f"+{row.plus:.6g}"
            )

        if self.autocorrelation_time is None:
            lines.extend(
                [
                    "",
                    (
                        "Autocorrelation time: "
                        "not reliably available"
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Autocorrelation times:",
                ]
            )

            for name, value in zip(
                self.free_parameter_names,
                self.autocorrelation_time,
                strict=True,
            ):
                lines.append(
                    f"  {name}: {value:.4f}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"BayesianResult("
            f"n_walkers={self.n_walkers}, "
            f"n_steps={self.n_steps}, "
            f"n_parameters={self.n_parameters}, "
            f"mean_acceptance_fraction="
            f"{self.mean_acceptance_fraction:.4f}"
            f")"
        )
