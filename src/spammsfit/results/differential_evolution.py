"""Result container for differential-evolution fitting."""

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


class DEResult(BaseResult):
    """
    Store the result of a differential-evolution fit.

    Parameters
    ----------
    best_vector
        Best free-parameter vector returned by differential evolution.
    best_parameters
        Complete fixed and fitted parameter mapping.
    free_parameter_names
        Names corresponding to entries in best_vector.
    population
        Final differential-evolution population.
    population_energies
        Objective values for the final population.
    generation_history
        Best solution and convergence information after each generation.
    settings
        Differential-evolution configuration.
    success
        Whether SciPy reported successful convergence.
    message
        SciPy termination message.
    n_iterations
        Number of completed DE generations.
    reported_n_evaluations
        Number of objective evaluations reported by SciPy.
    n_evaluations
        Number of actual SPAMMS model evaluations, including any final
        detailed best-model calculation.
    runtime
        Total fitting runtime in seconds.
    evaluation
        Detailed final evaluation of the best solution.
    metadata
        Optional additional metadata.

    Notes
    -----
    The final DE population is not a posterior distribution and is not
    interpreted as a parameter uncertainty.
    """

    def __init__(
        self,
        *,
        best_vector: ArrayLike,
        best_parameters: Mapping[str, float | int],
        free_parameter_names: tuple[str, ...],
        population: ArrayLike,
        population_energies: ArrayLike,
        generation_history: pd.DataFrame,
        settings: Mapping[str, Any],
        success: bool,
        message: str,
        n_iterations: int,
        reported_n_evaluations: int,
        n_evaluations: int,
        runtime: float,
        evaluation: EvaluationDetails,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.best_vector = np.asarray(
            best_vector,
            dtype=np.float64,
        ).copy()

        self.population = np.asarray(
            population,
            dtype=np.float64,
        ).copy()

        self.population_energies = np.asarray(
            population_energies,
            dtype=np.float64,
        ).copy()

        self.generation_history = (
            self._prepare_generation_history(
                generation_history
            )
        )

        self.settings = dict(settings)
        self.success = bool(success)
        self.message = str(message)
        self.n_iterations = int(
            n_iterations
        )

        self.reported_n_evaluations = int(
            reported_n_evaluations
        )

        self._validate_de_arrays(
            free_parameter_names=(
                free_parameter_names
            )
        )

        combined_metadata = {
            "optimizer": (
                "scipy.optimize."
                "differential_evolution"
            ),
            "success": self.success,
            "message": self.message,
            "n_iterations": (
                self.n_iterations
            ),
            "reported_n_evaluations": (
                self.reported_n_evaluations
            ),
        }

        if metadata is not None:
            combined_metadata.update(
                dict(metadata)
            )

        super().__init__(
            method="Differential evolution",
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
    def _prepare_generation_history(
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and copy the DE generation history."""

        if not isinstance(
            history,
            pd.DataFrame,
        ):
            raise TypeError(
                "generation_history must be "
                "a pandas DataFrame."
            )

        required_columns = {
            "generation",
            "chi2",
            "convergence",
            "n_evaluations",
        }

        missing = (
            required_columns
            - set(history.columns)
        )

        if missing:
            raise ValueError(
                "generation_history is missing required "
                f"columns: {sorted(missing)}."
            )

        prepared = history.copy().reset_index(
            drop=True
        )

        if prepared.empty:
            return prepared

        generations = prepared[
            "generation"
        ].to_numpy(
            dtype=int
        )

        expected = np.arange(
            1,
            len(prepared) + 1,
            dtype=int,
        )

        if not np.array_equal(
            generations,
            expected,
        ):
            raise ValueError(
                "Generation numbers must begin at 1 "
                "and increase consecutively."
            )

        chi2_values = prepared[
            "chi2"
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(chi2_values)
        ):
            raise ValueError(
                "generation_history contains "
                "non-finite chi-square values."
            )

        return prepared

    def _validate_de_arrays(
        self,
        free_parameter_names: tuple[str, ...],
    ) -> None:
        """Validate DE-specific arrays and counters."""

        n_parameters = len(
            free_parameter_names
        )

        if n_parameters < 1:
            raise ValueError(
                "DEResult requires at least one "
                "free parameter."
            )

        if self.best_vector.ndim != 1:
            raise ValueError(
                "best_vector must be one-dimensional."
            )

        if self.best_vector.size != n_parameters:
            raise ValueError(
                "best_vector size does not match the "
                "number of free parameters."
            )

        if not np.all(
            np.isfinite(self.best_vector)
        ):
            raise ValueError(
                "best_vector contains non-finite values."
            )

        if self.population.ndim != 2:
            raise ValueError(
                "population must be two-dimensional."
            )

        if self.population.shape[1] != n_parameters:
            raise ValueError(
                "Population width does not match the "
                "number of free parameters."
            )

        if not np.all(
            np.isfinite(self.population)
        ):
            raise ValueError(
                "population contains non-finite values."
            )

        if self.population_energies.ndim != 1:
            raise ValueError(
                "population_energies must be "
                "one-dimensional."
            )

        if (
            self.population_energies.size
            != self.population.shape[0]
        ):
            raise ValueError(
                "population_energies size does not match "
                "the population size."
            )

        if not np.all(
            np.isfinite(
                self.population_energies
            )
        ):
            raise ValueError(
                "population_energies contains "
                "non-finite values."
            )

        if self.n_iterations < 0:
            raise ValueError(
                "n_iterations cannot be negative."
            )

        if self.reported_n_evaluations < 1:
            raise ValueError(
                "reported_n_evaluations must be "
                "at least 1."
            )

        self.best_vector.flags.writeable = False
        self.population.flags.writeable = False
        self.population_energies.flags.writeable = False

    @property
    def population_size(self) -> int:
        """Return the final population size."""

        return int(
            self.population.shape[0]
        )

    @property
    def n_parameters(self) -> int:
        """Return the number of fitted parameters."""

        return int(
            self.best_vector.size
        )

    @property
    def best_population_index(self) -> int:
        """Return the index of the lowest-energy population member."""

        return int(
            np.argmin(
                self.population_energies
            )
        )

    @property
    def best_population_vector(
        self,
    ) -> FloatArray:
        """Return the lowest-energy final population member."""

        return self.population[
            self.best_population_index
        ]

    @property
    def best_population_energy(self) -> float:
        """Return the lowest final population objective value."""

        return float(
            self.population_energies[
                self.best_population_index
            ]
        )

    def convergence_table(
        self,
    ) -> pd.DataFrame:
        """Return a copy of the generation history."""

        return self.generation_history.copy()

    def population_table(
        self,
    ) -> pd.DataFrame:
        """
        Return the final DE population as a labelled table.
        """

        population_table = pd.DataFrame(
            self.population,
            columns=self.free_parameter_names,
        )

        population_table["chi2"] = (
            self.population_energies
        )

        population_table = (
            population_table
            .sort_values(
                "chi2",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        population_table["population_rank"] = (
            np.arange(
                1,
                len(population_table) + 1,
                dtype=int,
            )
        )

        return population_table

    def parameter_history(
        self,
        parameter: str,
    ) -> pd.DataFrame:
        """
        Return the best value of one parameter by generation.
        """

        if parameter not in (
            self.free_parameter_names
        ):
            raise KeyError(
                f"Unknown free parameter "
                f"{parameter!r}. Available parameters: "
                f"{self.free_parameter_names}."
            )

        if parameter not in (
            self.generation_history.columns
        ):
            raise KeyError(
                f"Generation history does not contain "
                f"parameter {parameter!r}."
            )

        return self.generation_history[
            [
                "generation",
                parameter,
                "chi2",
                "convergence",
                "n_evaluations",
            ]
        ].copy()

    def to_dict(self) -> dict[str, Any]:
        """Return serializable scalar DE result information."""

        result = super().to_dict()

        result.update(
            {
                "success": self.success,
                "message": self.message,
                "n_iterations": (
                    self.n_iterations
                ),
                "reported_n_evaluations": (
                    self.reported_n_evaluations
                ),
                "population_size": (
                    self.population_size
                ),
                "best_vector": (
                    self.best_vector.tolist()
                ),
                "best_population_energy": (
                    self.best_population_energy
                ),
                "settings": dict(
                    self.settings
                ),
            }
        )

        return result

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """
        Save the differential-evolution result.

        The following files are created:

        - ``summary.json``
        - ``generation_history.csv``
        - ``final_population.csv``
        - ``de_arrays.npz``
        - ``best_model.npz``
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

        self.generation_history.to_csv(
            output_directory
            / "generation_history.csv",
            index=False,
        )

        self.population_table().to_csv(
            output_directory
            / "final_population.csv",
            index=False,
        )

        np.savez_compressed(
            output_directory
            / "de_arrays.npz",
            best_vector=self.best_vector,
            population=self.population,
            population_energies=(
                self.population_energies
            ),
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
        """Return a readable DE result summary."""

        base_summary = super().summary()

        de_summary = (
            "\nDifferential-evolution information:\n"
            f"  Success: {self.success}\n"
            f"  Message: {self.message}\n"
            f"  Generations: "
            f"{self.n_iterations}\n"
            f"  SciPy evaluations: "
            f"{self.reported_n_evaluations}\n"
            f"  Population size: "
            f"{self.population_size}\n"
            "  Parameter intervals: not available "
            "from differential evolution alone"
        )

        return base_summary + de_summary

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"DEResult("
            f"success={self.success}, "
            f"chi2={self.chi2:.6f}, "
            f"n_iterations={self.n_iterations}, "
            f"population_size="
            f"{self.population_size}"
            f")"
        )
