"""Differential-evolution fitting for SPAMMSFit."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from spammsfit.core import SpammsFit
from spammsfit.fits.base import BaseFit
from spammsfit.results.differential_evolution import (
    DEResult,
)


class DEFit(BaseFit):
    """
    Fit continuous SPAMMS parameters using differential evolution.

    Parameters
    ----------
    spamms_fit
        Configured SPAMMSFit calculation.

    Notes
    -----
    The free parameters and their bounds are obtained directly from the
    associated ParameterSet.
    """

    method_name = "Differential evolution"

    def __init__(
        self,
        spamms_fit: SpammsFit,
    ) -> None:
        super().__init__(
            spamms_fit=spamms_fit,
        )

        self._n_objective_requests = 0
        self._best_chi2 = np.inf
        self._best_vector: np.ndarray | None = None
        self._generation_records: list[
            dict[str, Any]
        ] = []

        self._progress_bar: Any | None = None

    def run(
        self,
        *,
        strategy: str = "best1bin",
        maxiter: int = 30,
        popsize: int = 8,
        tol: float = 0.01,
        atol: float = 0.0,
        mutation: float | tuple[float, float] = (
            0.5,
            1.0,
        ),
        recombination: float = 0.7,
        seed: int | None = None,
        initial_guess: ArrayLike | None = None,
        init: str | ArrayLike = "latinhypercube",
        polish: bool = False,
        progress: bool = True,
        generation_output: bool = True,
        workers: int = 1,
    ) -> DEResult:
        """
        Run differential evolution.

        Parameters
        ----------
        strategy
            SciPy differential-evolution strategy.
        maxiter
            Maximum number of generations.
        popsize
            Population multiplier. The nominal population size is
            ``popsize * number_of_free_parameters``.
        tol
            Relative convergence tolerance.
        atol
            Absolute convergence tolerance.
        mutation
            Mutation constant or dithering range.
        recombination
            Recombination probability.
        seed
            Random-number seed.
        initial_guess
            Optional initial free-parameter vector.
        init
            SciPy population-initialization method or an explicit
            population array.
        polish
            Apply SciPy's local polishing stage after DE.
        progress
            Display a model-evaluation progress bar.
        generation_output
            Print the best solution after each generation.
        workers
            Number of DE workers. The initial implementation currently
            supports only ``workers=1``.

        Returns
        -------
        DEResult
            Completed differential-evolution result.
        """

        self._start_run()
        run_start = time.perf_counter()

        try:
            self.require_continuous_parameters()

            settings = self._prepare_settings(
                strategy=strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                atol=atol,
                mutation=mutation,
                recombination=recombination,
                seed=seed,
                initial_guess=initial_guess,
                init=init,
                polish=polish,
                progress=progress,
                generation_output=(
                    generation_output
                ),
                workers=workers,
            )

            free_names = (
                self.spamms_fit.parameters.free_names()
            )

            bounds = list(
                self.spamms_fit.parameters.continuous_bounds()
            )

            n_parameters = len(free_names)

            population_size = (
                settings["popsize"]
                * n_parameters
            )

            maximum_evaluations = (
                population_size
                * (
                    settings["maxiter"] + 1
                )
            )

            self._reset_tracking()

            starting_model_count = (
                self.spamms_fit.n_model_evaluations
            )

            if settings["progress"]:
                self._progress_bar = tqdm(
                    total=maximum_evaluations,
                    desc="DE SPAMMS fit",
                    unit="model",
                )

            scipy_arguments: dict[
                str,
                Any,
            ] = {
                "func": self._objective,
                "bounds": bounds,
                "strategy": (
                    settings["strategy"]
                ),
                "maxiter": (
                    settings["maxiter"]
                ),
                "popsize": (
                    settings["popsize"]
                ),
                "tol": settings["tol"],
                "atol": settings["atol"],
                "mutation": (
                    settings["mutation"]
                ),
                "recombination": (
                    settings["recombination"]
                ),
                "seed": settings["seed"],
                "callback": (
                    self._generation_callback
                ),
                "disp": False,
                "polish": (
                    settings["polish"]
                ),
                "init": init,
                "updating": "immediate",
                "workers": 1,
            }

            if (
                settings["initial_guess"]
                is not None
            ):
                scipy_arguments["x0"] = (
                    settings["initial_guess"]
                )

            try:
                scipy_result = (
                    differential_evolution(
                        **scipy_arguments
                    )
                )
            finally:
                if self._progress_bar is not None:
                    self._progress_bar.close()
                    self._progress_bar = None

            best_vector = np.asarray(
                scipy_result.x,
                dtype=np.float64,
            )

            optimizer_best_chi2 = float(
                scipy_result.fun
            )

            # One final detailed evaluation is required for the model,
            # per-line chi-square values and residuals stored by
            # DEResult.
            evaluation = self.spamms_fit.evaluate(
                best_vector
            )

            best_parameters = (
                evaluation["parameters"]
            )

            n_model_evaluations = (
                self.spamms_fit.n_model_evaluations
                - starting_model_count
            )

            generation_history = pd.DataFrame(
                self._generation_records
            )

            runtime = (
                time.perf_counter()
                - run_start
            )

            result = DEResult(
                best_vector=best_vector,
                best_parameters=best_parameters,
                free_parameter_names=(
                    free_names
                ),
                population=(
                    scipy_result.population
                ),
                population_energies=(
                    scipy_result.population_energies
                ),
                generation_history=(
                    generation_history
                ),
                settings={
                    **self._serializable_settings(
                        settings
                    ),
                    "bounds": [
                        list(bound)
                        for bound in bounds
                    ],
                    "population_size": (
                        population_size
                    ),
                    "maximum_evaluations": (
                        maximum_evaluations
                    ),
                    "optimizer_best_chi2": (
                        optimizer_best_chi2
                    ),
                    "final_evaluation_chi2": (
                        evaluation["chi2"]
                    ),
                },
                success=bool(
                    scipy_result.success
                ),
                message=str(
                    scipy_result.message
                ),
                n_iterations=int(
                    scipy_result.nit
                ),
                reported_n_evaluations=int(
                    scipy_result.nfev
                ),
                n_evaluations=(
                    n_model_evaluations
                ),
                runtime=runtime,
                evaluation=evaluation,
                metadata={
                    "objective_requests": (
                        self._n_objective_requests
                    ),
                    "selected_lines": list(
                        self.spamms_fit.spectrum.line_names
                    ),
                },
            )

        except Exception as error:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None

            self._fail_run(error)
            raise

        return self._finish_run(result)

    def _objective(
        self,
        theta: ArrayLike,
    ) -> float:
        """
        Calculate multiline chi-square for one DE proposal.
        """

        theta_array = np.asarray(
            theta,
            dtype=np.float64,
        )

        self._n_objective_requests += 1

        chi2_value = self.spamms_fit.chi2(
            theta_array
        )

        if self._progress_bar is not None:
            self._progress_bar.update(1)

        if chi2_value < self._best_chi2:
            self._best_chi2 = chi2_value
            self._best_vector = (
                theta_array.copy()
            )

            self._update_progress_postfix()

        return chi2_value

    def _generation_callback(
        self,
        best_vector: ArrayLike,
        convergence: float,
    ) -> bool:
        """
        Record the best solution after one DE generation.

        Returning False instructs SciPy to continue.
        """

        generation_number = (
            len(self._generation_records) + 1
        )

        callback_vector = np.asarray(
            best_vector,
            dtype=np.float64,
        )

        # The objective tracker normally contains the same or a better
        # vector. The callback vector is used as a safe fallback.
        if self._best_vector is None:
            tracked_vector = (
                callback_vector.copy()
            )
        else:
            tracked_vector = (
                self._best_vector.copy()
            )

        record: dict[str, Any] = {
            "generation": (
                generation_number
            ),
            "chi2": float(
                self._best_chi2
            ),
            "convergence": float(
                convergence
            ),
            "n_evaluations": (
                self._n_objective_requests
            ),
        }

        for name, value in zip(
            self.spamms_fit.parameters.free_names(),
            tracked_vector,
            strict=True,
        ):
            record[name] = float(value)

        self._generation_records.append(
            record
        )

        if self._current_generation_output:
            parameter_text = ", ".join(
                f"{name}={value:.6g}"
                for name, value in zip(
                    self.spamms_fit.parameters.free_names(),
                    tracked_vector,
                    strict=True,
                )
            )

            print(
                f"\nGeneration "
                f"{generation_number}: "
                f"chi2={self._best_chi2:.6f}, "
                f"{parameter_text}, "
                f"convergence="
                f"{convergence:.6e}"
            )

        return False

    def _update_progress_postfix(self) -> None:
        """Display the current best solution on the progress bar."""

        if (
            self._progress_bar is None
            or self._best_vector is None
        ):
            return

        postfix: dict[str, str] = {
            "chi2": (
                f"{self._best_chi2:.3f}"
            )
        }

        for name, value in zip(
            self.spamms_fit.parameters.free_names(),
            self._best_vector,
            strict=True,
        ):
            postfix[name] = f"{value:.4g}"

        self._progress_bar.set_postfix(
            postfix
        )

    def _prepare_settings(
        self,
        *,
        strategy: str,
        maxiter: int,
        popsize: int,
        tol: float,
        atol: float,
        mutation: float | tuple[float, float],
        recombination: float,
        seed: int | None,
        initial_guess: ArrayLike | None,
        init: str | ArrayLike,
        polish: bool,
        progress: bool,
        generation_output: bool,
        workers: int,
    ) -> dict[str, Any]:
        """Validate and normalize DE settings."""

        strategy = str(strategy).strip()

        if not strategy:
            raise ValueError(
                "strategy cannot be empty."
            )

        maxiter = int(maxiter)
        popsize = int(popsize)
        workers = int(workers)

        if maxiter < 0:
            raise ValueError(
                "maxiter cannot be negative."
            )

        if popsize < 1:
            raise ValueError(
                "popsize must be at least 1."
            )

        if workers != 1:
            raise NotImplementedError(
                "The initial DEFit implementation supports "
                "workers=1 only. Parallel DE requires "
                "process-safe progress tracking and runner "
                "statistics and will be added after the "
                "serial implementation is verified."
            )

        tol = float(tol)
        atol = float(atol)

        if tol < 0.0:
            raise ValueError(
                "tol cannot be negative."
            )

        if atol < 0.0:
            raise ValueError(
                "atol cannot be negative."
            )

        recombination = float(
            recombination
        )

        if not 0.0 <= recombination <= 1.0:
            raise ValueError(
                "recombination must lie between "
                "0 and 1."
            )

        prepared_mutation = (
            self._prepare_mutation(
                mutation
            )
        )

        if seed is not None:
            seed = int(seed)

        prepared_initial_guess = (
            self._prepare_initial_guess(
                initial_guess
            )
        )

        if not isinstance(init, str):
            init_array = np.asarray(
                init,
                dtype=np.float64,
            )

            if init_array.ndim != 2:
                raise ValueError(
                    "An explicit DE initial population "
                    "must be two-dimensional."
                )

            if (
                init_array.shape[1]
                != self.spamms_fit.parameters.n_free
            ):
                raise ValueError(
                    "Initial population width does not "
                    "match the number of free parameters."
                )

            init = init_array

        self._current_generation_output = bool(
            generation_output
        )

        return {
            "strategy": strategy,
            "maxiter": maxiter,
            "popsize": popsize,
            "tol": tol,
            "atol": atol,
            "mutation": prepared_mutation,
            "recombination": recombination,
            "seed": seed,
            "initial_guess": (
                prepared_initial_guess
            ),
            "init": init,
            "polish": bool(polish),
            "progress": bool(progress),
            "generation_output": bool(
                generation_output
            ),
            "workers": workers,
        }

    @staticmethod
    def _prepare_mutation(
        mutation: float | tuple[float, float],
    ) -> float | tuple[float, float]:
        """Validate the mutation constant or dithering range."""

        if isinstance(
            mutation,
            Sequence,
        ) and not isinstance(
            mutation,
            (str, bytes),
        ):
            if len(mutation) != 2:
                raise ValueError(
                    "A mutation range must contain "
                    "exactly two values."
                )

            lower = float(mutation[0])
            upper = float(mutation[1])

            if not (
                0.0 <= lower < upper < 2.0
            ):
                raise ValueError(
                    "Mutation dithering limits must "
                    "satisfy 0 <= lower < upper < 2."
                )

            return lower, upper

        value = float(mutation)

        if not 0.0 <= value < 2.0:
            raise ValueError(
                "mutation must satisfy "
                "0 <= mutation < 2."
            )

        return value

    def _prepare_initial_guess(
        self,
        initial_guess: ArrayLike | None,
    ) -> np.ndarray | None:
        """Validate an optional initial parameter vector."""

        if initial_guess is None:
            return None

        guess = np.asarray(
            initial_guess,
            dtype=np.float64,
        )

        if guess.ndim != 1:
            raise ValueError(
                "initial_guess must be "
                "one-dimensional."
            )

        if (
            guess.size
            != self.spamms_fit.parameters.n_free
        ):
            raise ValueError(
                "initial_guess size does not match "
                "the number of free parameters."
            )

        if not (
            self.spamms_fit.parameters.vector_in_bounds(
                guess
            )
        ):
            raise ValueError(
                "initial_guess must lie strictly "
                "inside all parameter bounds."
            )

        return guess

    @staticmethod
    def _serializable_settings(
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert DE settings to JSON-compatible values."""

        serializable = dict(settings)

        initial_guess = serializable[
            "initial_guess"
        ]

        if initial_guess is not None:
            serializable["initial_guess"] = (
                initial_guess.tolist()
            )

        init = serializable["init"]

        if isinstance(init, np.ndarray):
            serializable["init"] = (
                init.tolist()
            )

        mutation = serializable[
            "mutation"
        ]

        if isinstance(mutation, tuple):
            serializable["mutation"] = (
                list(mutation)
            )

        return serializable

    def _reset_tracking(self) -> None:
        """Reset objective and generation tracking."""

        self._n_objective_requests = 0
        self._best_chi2 = np.inf
        self._best_vector = None
        self._generation_records = []
        self._progress_bar = None

    @property
    def n_objective_requests(self) -> int:
        """Return the number of DE objective calls."""

        return self._n_objective_requests

    @property
    def generation_history(
        self,
    ) -> pd.DataFrame:
        """Return the current generation history."""

        return pd.DataFrame(
            self._generation_records
        )

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"DEFit("
            f"status={self.status!r}, "
            f"n_free="
            f"{self.spamms_fit.parameters.n_free}, "
            f"n_objective_requests="
            f"{self._n_objective_requests}"
            f")"
        )
