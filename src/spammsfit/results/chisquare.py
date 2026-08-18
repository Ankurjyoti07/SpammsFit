"""Result container for grid-based chi-square fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from spammsfit.core import EvaluationDetails
from spammsfit.results.base import BaseResult


VALID_SIGMA_MODES = {
    "shared",
    "per_line",
}


class ChiSquareResult(BaseResult):
    """
    Store results from a grid-based chi-square search.

    Parameters
    ----------
    global_table
        Ranked table containing the total chi-square for each valid
        global grid solution.
    line_table
        Table containing the line-by-line contributions associated with
        the global solutions.
    sigma_mode
        Either ``"shared"`` or ``"per_line"``.
    model_index
        Path to the model-index file used for the search.
    best_parameters
        Complete parameter description of the best solution.
    free_parameter_names
        Parameters explored by the grid search.
    runtime
        Total grid-search runtime in seconds.
    n_evaluations
        Number of individual model-profile comparisons.
    evaluation
        Detailed best-fitting model evaluation.
    metadata
        Optional additional information.
    """

    def __init__(
        self,
        *,
        global_table: pd.DataFrame,
        line_table: pd.DataFrame,
        sigma_mode: str,
        model_index: str | Path,
        best_parameters: Mapping[str, float | int],
        free_parameter_names: tuple[str, ...],
        runtime: float,
        n_evaluations: int,
        evaluation: EvaluationDetails,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.sigma_mode = str(
            sigma_mode
        ).strip()

        if self.sigma_mode not in VALID_SIGMA_MODES:
            raise ValueError(
                "sigma_mode must be either 'shared' "
                "or 'per_line'."
            )

        self.model_index = (
            Path(model_index)
            .expanduser()
            .resolve()
        )

        self.global_table = self._prepare_global_table(
            global_table
        )

        self.line_table = self._prepare_line_table(
            line_table
        )

        self._validate_tables()

        combined_metadata = {
            "sigma_mode": self.sigma_mode,
            "model_index": str(
                self.model_index
            ),
        }

        if metadata is not None:
            combined_metadata.update(
                dict(metadata)
            )

        super().__init__(
            method="Grid chi-square",
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
    def _prepare_global_table(
        table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate, copy and rank the global result table."""

        if not isinstance(
            table,
            pd.DataFrame,
        ):
            raise TypeError(
                "global_table must be a pandas DataFrame."
            )

        if table.empty:
            raise ValueError(
                "global_table cannot be empty."
            )

        if "chi2_total" not in table.columns:
            raise ValueError(
                "global_table must contain a "
                "'chi2_total' column."
            )

        prepared = (
            table.copy()
            .sort_values(
                "chi2_total",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        prepared["global_rank"] = np.arange(
            1,
            len(prepared) + 1,
            dtype=int,
        )

        minimum_chi2 = float(
            prepared.loc[0, "chi2_total"]
        )

        prepared["delta_chi2"] = (
            prepared["chi2_total"]
            - minimum_chi2
        )

        return prepared

    @staticmethod
    def _prepare_line_table(
        table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and copy the line-by-line result table."""

        if not isinstance(
            table,
            pd.DataFrame,
        ):
            raise TypeError(
                "line_table must be a pandas DataFrame."
            )

        if table.empty:
            raise ValueError(
                "line_table cannot be empty."
            )

        required_columns = {
            "line_name",
            "chi2_line",
            "n_pix_line",
            "profile_path",
        }

        missing = (
            required_columns
            - set(table.columns)
        )

        if missing:
            raise ValueError(
                "line_table is missing required columns: "
                f"{sorted(missing)}."
            )

        return table.copy().reset_index(
            drop=True
        )

    def _validate_tables(self) -> None:
        """Validate consistency between global and line tables."""

        chi2_values = self.global_table[
            "chi2_total"
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(chi2_values)
        ):
            raise ValueError(
                "global_table contains non-finite "
                "chi-square values."
            )

        if np.any(chi2_values < 0.0):
            raise ValueError(
                "global_table contains negative "
                "chi-square values."
            )

        line_chi2 = self.line_table[
            "chi2_line"
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(line_chi2)
        ):
            raise ValueError(
                "line_table contains non-finite "
                "chi-square values."
            )

        if np.any(line_chi2 < 0.0):
            raise ValueError(
                "line_table contains negative "
                "chi-square values."
            )

        n_pixels = self.line_table[
            "n_pix_line"
        ].to_numpy(
            dtype=int
        )

        if np.any(n_pixels < 1):
            raise ValueError(
                "Every line result must contain at least "
                "one fitted pixel."
            )

        if self.sigma_mode == "shared":
            required_global = {
                "model_id",
                "sigma_R",
                "sigma_T",
            }

            required_line = {
                "model_id",
                "sigma_R",
                "sigma_T",
            }

        else:
            required_global = {
                "teff",
                "r_pole",
                "mass",
                "inclination",
                "v_crit_frac",
            }

            required_line = {
                "teff",
                "r_pole",
                "mass",
                "inclination",
                "v_crit_frac",
                "sigma_R",
                "sigma_T",
            }

        missing_global = (
            required_global
            - set(self.global_table.columns)
        )

        missing_line = (
            required_line
            - set(self.line_table.columns)
        )

        if missing_global:
            raise ValueError(
                "global_table is missing columns required "
                f"for {self.sigma_mode!r} mode: "
                f"{sorted(missing_global)}."
            )

        if missing_line:
            raise ValueError(
                "line_table is missing columns required "
                f"for {self.sigma_mode!r} mode: "
                f"{sorted(missing_line)}."
            )

    @property
    def best_global_row(
        self,
    ) -> pd.Series:
        """Return the highest-ranked global solution."""

        return self.global_table.iloc[0].copy()

    @property
    def best_line_rows(
        self,
    ) -> pd.DataFrame:
        """
        Return line records associated with the best global solution.
        """

        if "global_rank" not in self.line_table.columns:
            raise ValueError(
                "line_table does not contain global ranks."
            )

        return (
            self.line_table[
                self.line_table["global_rank"] == 1
            ]
            .copy()
            .reset_index(drop=True)
        )

    @property
    def minimum_chi2(self) -> float:
        """Return the minimum total chi-square."""

        return float(
            self.global_table.loc[
                0,
                "chi2_total",
            ]
        )

    @property
    def n_global_solutions(self) -> int:
        """Return the number of valid global grid solutions."""

        return len(self.global_table)

    @property
    def n_line_records(self) -> int:
        """Return the number of stored line-level records."""

        return len(self.line_table)

    def top(
        self,
        n: int = 10,
    ) -> pd.DataFrame:
        """Return the highest-ranked global solutions."""

        n = int(n)

        if n < 1:
            raise ValueError(
                "n must be at least 1."
            )

        return self.global_table.head(
            n
        ).copy()

    def line_results(
        self,
        line_name: str,
    ) -> pd.DataFrame:
        """Return all stored results for one spectral line."""

        selected = self.line_table[
            self.line_table["line_name"]
            == line_name
        ]

        if selected.empty:
            available = sorted(
                self.line_table[
                    "line_name"
                ].unique()
            )

            raise KeyError(
                f"No results are available for line "
                f"{line_name!r}. Available lines: "
                f"{available}."
            )

        return selected.copy().reset_index(
            drop=True
        )

    def profile_likelihood(
        self,
        parameter: str,
    ) -> pd.DataFrame:
        """
        Return minimum chi-square as a function of one grid parameter.

        For every unique parameter value, all other grid dimensions are
        profiled out by retaining the minimum total chi-square.
        """

        if parameter not in self.global_table.columns:
            raise KeyError(
                f"Parameter {parameter!r} is not present "
                "in the global result table."
            )

        profile = (
            self.global_table
            .groupby(
                parameter,
                as_index=False,
                sort=True,
            )["chi2_total"]
            .min()
        )

        profile["delta_chi2"] = (
            profile["chi2_total"]
            - self.minimum_chi2
        )

        return profile

    def parameter_map(
        self,
        x_parameter: str,
        y_parameter: str,
    ) -> pd.DataFrame:
        """
        Return a two-dimensional profile chi-square table.

        All other grid dimensions are profiled out.
        """

        for parameter in (
            x_parameter,
            y_parameter,
        ):
            if parameter not in (
                self.global_table.columns
            ):
                raise KeyError(
                    f"Parameter {parameter!r} is not "
                    "present in the global result table."
                )

        parameter_map = (
            self.global_table
            .groupby(
                [
                    x_parameter,
                    y_parameter,
                ],
                as_index=False,
                sort=True,
            )["chi2_total"]
            .min()
        )

        parameter_map["delta_chi2"] = (
            parameter_map["chi2_total"]
            - self.minimum_chi2
        )

        return parameter_map

    def to_dict(self) -> dict[str, Any]:
        """Return serializable scalar result information."""

        result = super().to_dict()

        result.update(
            {
                "sigma_mode": self.sigma_mode,
                "model_index": str(
                    self.model_index
                ),
                "n_global_solutions": (
                    self.n_global_solutions
                ),
                "n_line_records": (
                    self.n_line_records
                ),
            }
        )

        return result

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """
        Save chi-square results and best-model arrays.

        The following files are created:

        - ``summary.json``
        - ``chisq_results_global.csv``
        - ``chisq_results_lines.csv``
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

        summary_file = (
            output_directory
            / "summary.json"
        )

        summary_file.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.global_table.to_csv(
            output_directory
            / "chisq_results_global.csv",
            index=False,
        )

        self.line_table.to_csv(
            output_directory
            / "chisq_results_lines.csv",
            index=False,
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
        """Return a readable grid-search summary."""

        base_summary = super().summary()

        grid_summary = (
            "\nGrid-search information:\n"
            f"  Sigma mode: {self.sigma_mode}\n"
            f"  Model index: {self.model_index}\n"
            f"  Global solutions: "
            f"{self.n_global_solutions}\n"
            f"  Line records: "
            f"{self.n_line_records}"
        )

        return base_summary + grid_summary

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"ChiSquareResult("
            f"sigma_mode={self.sigma_mode!r}, "
            f"chi2={self.chi2:.6f}, "
            f"n_global_solutions="
            f"{self.n_global_solutions}, "
            f"n_evaluations="
            f"{self.n_evaluations}"
            f")"
        )
