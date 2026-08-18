"""Grid-based chi-square fitting for SPAMMSFit."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spammsfit.core import (
    EvaluationDetails,
    SpammsFit,
)
from spammsfit.fits.base import BaseFit
from spammsfit.forward.output import (
    ModelSpectra,
)
from spammsfit.likelihood.interpolation import (
    interpolate_model,
)
from spammsfit.likelihood.statistics import (
    chi_square,
    reduced_chi_square,
)
from spammsfit.results.chisquare import (
    ChiSquareResult,
)


VALID_SIGMA_MODES = {
    "shared",
    "per_line",
}

GLOBAL_PARAMETERS = (
    "teff",
    "r_pole",
    "mass",
    "inclination",
    "v_crit_frac",
)

SIGMA_PARAMETERS = (
    "sigma_R",
    "sigma_T",
)

GRID_PARAMETERS = (
    *GLOBAL_PARAMETERS,
    *SIGMA_PARAMETERS,
)

REQUIRED_INDEX_COLUMNS = {
    "model_id",
    "line_name",
    "profile_path",
    "profile_exists",
    *GRID_PARAMETERS,
}


class ChiSquareFit(BaseFit):
    """
    Perform chi-square fitting on a precomputed SPAMMS grid.

    Parameters
    ----------
    spamms_fit
        Configured common SPAMMSFit calculation.
    model_index
        Path to the model-index Parquet file.
    sigma_mode
        Treatment of sigma_R and sigma_T:

        ``"shared"``
            One sigma_R and sigma_T pair is shared by all fitted lines.

        ``"per_line"``
            Each fitted line independently selects its best sigma_R and
            sigma_T at every shared global grid point.

    Notes
    -----
    This class does not execute SPAMMS. It reads existing model profiles
    listed in the model index.
    """

    method_name = "Grid chi-square"

    def __init__(
        self,
        spamms_fit: SpammsFit,
        model_index: str | Path,
        *,
        sigma_mode: str = "shared",
    ) -> None:
        super().__init__(
            spamms_fit=spamms_fit,
        )

        self.model_index = (
            Path(model_index)
            .expanduser()
            .resolve()
        )

        if not self.model_index.is_file():
            raise FileNotFoundError(
                f"Model index not found: "
                f"{self.model_index}"
            )

        self.sigma_mode = str(
            sigma_mode
        ).strip()

        if self.sigma_mode not in VALID_SIGMA_MODES:
            raise ValueError(
                "sigma_mode must be either "
                "'shared' or 'per_line'."
            )

        self._n_profile_evaluations = 0

    def run(self) -> ChiSquareResult:
        """
        Execute the grid-based chi-square search.

        Returns
        -------
        ChiSquareResult
            Ranked global and line-level grid results.
        """

        self._start_run()
        calculation_start = time.perf_counter()

        try:
            self._validate_parameter_configuration()

            model_index = self._read_model_index()

            initial_rows = len(model_index)
            initial_models = (
                model_index["model_id"].nunique()
            )

            filtered_index = self._filter_index(
                model_index
            )

            if filtered_index.empty:
                raise ValueError(
                    "No grid models remain after applying "
                    "the selected lines and parameter constraints."
                )

            self._n_profile_evaluations = 0

            if self.sigma_mode == "shared":
                global_table, line_table = (
                    self._fit_shared_sigma(
                        filtered_index
                    )
                )
            else:
                global_table, line_table = (
                    self._fit_per_line_sigma(
                        filtered_index
                    )
                )

            if global_table.empty:
                raise RuntimeError(
                    "The grid search produced no valid "
                    "global solutions."
                )

            if line_table.empty:
                raise RuntimeError(
                    "The grid search produced no valid "
                    "line-level solutions."
                )

            global_table, line_table = (
                self._add_global_ranks(
                    global_table=global_table,
                    line_table=line_table,
                )
            )

            best_parameters = (
                self._build_best_parameters(
                    global_table=global_table,
                    line_table=line_table,
                )
            )

            evaluation = self._build_best_evaluation(
                best_parameters=best_parameters,
                global_table=global_table,
                line_table=line_table,
            )

            runtime = (
                time.perf_counter()
                - calculation_start
            )

            result = ChiSquareResult(
                global_table=global_table,
                line_table=line_table,
                sigma_mode=self.sigma_mode,
                model_index=self.model_index,
                best_parameters=best_parameters,
                free_parameter_names=(
                    self._result_free_parameter_names()
                ),
                runtime=runtime,
                n_evaluations=(
                    self._n_profile_evaluations
                ),
                evaluation=evaluation,
                metadata={
                    "selected_lines": list(
                        self.spamms_fit.spectrum.line_names
                    ),
                    "initial_index_rows": (
                        initial_rows
                    ),
                    "initial_unique_models": (
                        initial_models
                    ),
                    "filtered_index_rows": (
                        len(filtered_index)
                    ),
                    "filtered_unique_models": (
                        filtered_index[
                            "model_id"
                        ].nunique()
                    ),
                },
            )

        except Exception as error:
            self._fail_run(error)
            raise

        return self._finish_run(result)

    def _validate_parameter_configuration(
        self,
    ) -> None:
        """Validate parameters for a discrete grid search."""

        self.require_free_parameters()

        continuous_parameters = [
            parameter.name
            for parameter
            in self.spamms_fit.parameters.free_parameters()
            if not parameter.is_discrete
        ]

        if continuous_parameters:
            raise ValueError(
                "ChiSquareFit requires free parameters "
                "to have discrete permitted values. "
                "Continuous parameters found: "
                f"{continuous_parameters}."
            )

    def _read_model_index(
        self,
    ) -> pd.DataFrame:
        """Read and validate the model-index table."""

        try:
            model_index = pd.read_parquet(
                self.model_index
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not read model index: "
                f"{self.model_index}"
            ) from error

        missing = (
            REQUIRED_INDEX_COLUMNS
            - set(model_index.columns)
        )

        if missing:
            raise ValueError(
                "Model index is missing required columns: "
                f"{sorted(missing)}."
            )

        if model_index.empty:
            raise ValueError(
                "The model index is empty."
            )

        return model_index

    def _filter_index(
        self,
        model_index: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply selected-line, fixed-parameter and discrete-value filters.
        """

        selected_lines = (
            self.spamms_fit.spectrum.line_names
        )

        filtered = model_index[
            model_index["line_name"].isin(
                selected_lines
            )
        ].copy()

        for parameter_name in GRID_PARAMETERS:
            parameter = (
                self.spamms_fit.parameters[
                    parameter_name
                ]
            )

            if parameter.fixed:
                filtered = filtered[
                    np.isclose(
                        filtered[parameter_name],
                        parameter.value,
                    )
                ]

            elif parameter.is_discrete:
                if parameter.values is None:
                    raise RuntimeError(
                        f"Discrete parameter "
                        f"{parameter_name!r} has no "
                        "permitted values."
                    )

                permitted_values = np.asarray(
                    parameter.values
                )

                keep = np.zeros(
                    len(filtered),
                    dtype=bool,
                )

                for value in permitted_values:
                    keep |= np.isclose(
                        filtered[parameter_name],
                        value,
                    )

                filtered = filtered[keep]

        return filtered.reset_index(
            drop=True
        )

    def _fit_shared_sigma(
        self,
        model_index: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit one shared sigma_R and sigma_T pair across all lines.
        """

        selected_lines = (
            self.spamms_fit.spectrum.line_names
        )

        global_records: list[
            dict[str, Any]
        ] = []

        line_records: list[
            dict[str, Any]
        ] = []

        model_groups = model_index.groupby(
            "model_id",
            sort=False,
        )

        for model_id, model_group in model_groups:
            first = model_group.iloc[0]

            total_chi2 = 0.0
            total_pixels = 0
            valid_model = True

            current_line_records: list[
                dict[str, Any]
            ] = []

            for line_name in selected_lines:
                line_rows = model_group[
                    model_group["line_name"]
                    == line_name
                ]

                if len(line_rows) != 1:
                    valid_model = False
                    break

                row = line_rows.iloc[0]

                if not bool(
                    row["profile_exists"]
                ):
                    valid_model = False
                    break

                line_chi2, n_pixels = (
                    self._compare_profile(
                        line_name=line_name,
                        profile_path=row[
                            "profile_path"
                        ],
                    )
                )

                total_chi2 += line_chi2
                total_pixels += n_pixels

                current_line_records.append(
                    {
                        "model_id": model_id,
                        "line_name": line_name,
                        "chi2_line": line_chi2,
                        "n_pix_line": n_pixels,
                        "sigma_R": self._scalar(
                            row["sigma_R"]
                        ),
                        "sigma_T": self._scalar(
                            row["sigma_T"]
                        ),
                        "profile_path": str(
                            row["profile_path"]
                        ),
                    }
                )

            if not valid_model:
                continue

            n_free = (
                self._effective_n_free()
            )

            reduced = reduced_chi_square(
                chi2=total_chi2,
                n_pixels=total_pixels,
                n_free_parameters=n_free,
            )

            global_records.append(
                {
                    "model_id": model_id,
                    "chi2_total": total_chi2,
                    "red_chi2": reduced,
                    "n_pix_total": total_pixels,
                    "teff": self._scalar(
                        first["teff"]
                    ),
                    "r_pole": self._scalar(
                        first["r_pole"]
                    ),
                    "mass": self._scalar(
                        first["mass"]
                    ),
                    "inclination": self._scalar(
                        first["inclination"]
                    ),
                    "v_crit_frac": self._scalar(
                        first["v_crit_frac"]
                    ),
                    "sigma_R": self._scalar(
                        first["sigma_R"]
                    ),
                    "sigma_T": self._scalar(
                        first["sigma_T"]
                    ),
                }
            )

            line_records.extend(
                current_line_records
            )

        return (
            self._make_global_table(
                global_records
            ),
            pd.DataFrame(line_records),
        )

    def _fit_per_line_sigma(
        self,
        model_index: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit independent sigma_R and sigma_T values for every line.
        """

        selected_lines = (
            self.spamms_fit.spectrum.line_names
        )

        global_records: list[
            dict[str, Any]
        ] = []

        line_records: list[
            dict[str, Any]
        ] = []

        grouped = model_index.groupby(
            list(GLOBAL_PARAMETERS),
            sort=False,
            dropna=False,
        )

        for global_values, global_group in grouped:
            global_parameters = {
                name: self._scalar(value)
                for name, value in zip(
                    GLOBAL_PARAMETERS,
                    global_values,
                    strict=True,
                )
            }

            total_chi2 = 0.0
            total_pixels = 0
            valid_global = True

            current_line_records: list[
                dict[str, Any]
            ] = []

            for line_name in selected_lines:
                line_rows = global_group[
                    global_group["line_name"]
                    == line_name
                ]

                if line_rows.empty:
                    valid_global = False
                    break

                best_record: (
                    dict[str, Any] | None
                ) = None

                for row in line_rows.itertuples(
                    index=False
                ):
                    if not bool(
                        row.profile_exists
                    ):
                        continue

                    line_chi2, n_pixels = (
                        self._compare_profile(
                            line_name=line_name,
                            profile_path=(
                                row.profile_path
                            ),
                        )
                    )

                    if (
                        best_record is None
                        or line_chi2
                        < best_record[
                            "chi2_line"
                        ]
                    ):
                        best_record = {
                            **global_parameters,
                            "line_name": line_name,
                            "model_id": (
                                row.model_id
                            ),
                            "chi2_line": line_chi2,
                            "n_pix_line": n_pixels,
                            "sigma_R": self._scalar(
                                row.sigma_R
                            ),
                            "sigma_T": self._scalar(
                                row.sigma_T
                            ),
                            "profile_path": str(
                                row.profile_path
                            ),
                        }

                if best_record is None:
                    valid_global = False
                    break

                total_chi2 += best_record[
                    "chi2_line"
                ]

                total_pixels += best_record[
                    "n_pix_line"
                ]

                current_line_records.append(
                    best_record
                )

            if not valid_global:
                continue

            reduced = reduced_chi_square(
                chi2=total_chi2,
                n_pixels=total_pixels,
                n_free_parameters=(
                    self._effective_n_free()
                ),
            )

            global_records.append(
                {
                    **global_parameters,
                    "chi2_total": total_chi2,
                    "red_chi2": reduced,
                    "n_pix_total": total_pixels,
                }
            )

            line_records.extend(
                current_line_records
            )

        return (
            self._make_global_table(
                global_records
            ),
            pd.DataFrame(line_records),
        )

    @staticmethod
    def _make_global_table(
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Create and rank a global result table."""

        table = pd.DataFrame(records)

        if table.empty:
            return table

        return (
            table.sort_values(
                "chi2_total",
                kind="stable",
            )
            .reset_index(drop=True)
        )

    def _compare_profile(
        self,
        line_name: str,
        profile_path: str | Path,
    ) -> tuple[float, int]:
        """Compare one stored grid profile with one observed line."""

        profile_path = self._resolve_profile_path(
            profile_path
        )

        model_wave, model_flux = (
            self._read_model_profile(
                profile_path
            )
        )

        observed = (
            self.spamms_fit.spectrum.get_line(
                line_name
            )
        )

        interpolated_flux = interpolate_model(
            model_wavelength=model_wave,
            model_flux=model_flux,
            observed_wavelength=(
                observed.wavelength
            ),
            extrapolate=(
                self.spamms_fit.extrapolate
            ),
        )

        line_chi2 = chi_square(
            observed_flux=observed.flux,
            model_flux=interpolated_flux,
            uncertainty=observed.uncertainty,
        )

        self._n_profile_evaluations += 1

        return (
            line_chi2,
            observed.wavelength.size,
        )

    def _resolve_profile_path(
        self,
        profile_path: str | Path,
    ) -> Path:
        """Resolve an absolute or index-relative profile path."""

        profile_path = Path(
            profile_path
        ).expanduser()

        if not profile_path.is_absolute():
            profile_path = (
                self.model_index.parent
                / profile_path
            )

        profile_path = (
            profile_path.resolve()
        )

        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Grid profile not found: "
                f"{profile_path}"
            )

        return profile_path

    @staticmethod
    def _read_model_profile(
        profile_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one wavelength/flux grid profile."""

        try:
            model = np.loadtxt(
                profile_path,
                dtype=np.float64,
                usecols=(0, 1),
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not read grid profile: "
                f"{profile_path}"
            ) from error

        model = np.atleast_2d(model)

        if model.shape[0] < 2:
            raise ValueError(
                f"Grid profile contains fewer than "
                f"two pixels: {profile_path}"
            )

        model_wave = np.ascontiguousarray(
            model[:, 0],
            dtype=np.float64,
        )

        model_flux = np.ascontiguousarray(
            model[:, 1],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(model_wave)
        ):
            raise ValueError(
                f"Non-finite wavelengths in "
                f"{profile_path}."
            )

        if not np.all(
            np.isfinite(model_flux)
        ):
            raise ValueError(
                f"Non-finite fluxes in "
                f"{profile_path}."
            )

        if not np.all(
            np.diff(model_wave) > 0.0
        ):
            raise ValueError(
                f"Wavelengths are not strictly "
                f"increasing in {profile_path}."
            )

        model_wave.flags.writeable = False
        model_flux.flags.writeable = False

        return model_wave, model_flux

    def _add_global_ranks(
        self,
        global_table: pd.DataFrame,
        line_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add global ranks to global and line-level results."""

        global_table = (
            global_table
            .sort_values(
                "chi2_total",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        global_table["global_rank"] = (
            np.arange(
                1,
                len(global_table) + 1,
                dtype=int,
            )
        )

        if self.sigma_mode == "shared":
            rank_lookup = global_table[
                [
                    "model_id",
                    "global_rank",
                ]
            ]

            line_table = line_table.merge(
                rank_lookup,
                on="model_id",
                how="left",
                validate="many_to_one",
            )

        else:
            rank_lookup = global_table[
                [
                    *GLOBAL_PARAMETERS,
                    "global_rank",
                ]
            ]

            line_table = line_table.merge(
                rank_lookup,
                on=list(GLOBAL_PARAMETERS),
                how="left",
                validate="many_to_one",
            )

        line_table = (
            line_table.sort_values(
                [
                    "global_rank",
                    "line_name",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        return global_table, line_table

    def _build_best_parameters(
        self,
        global_table: pd.DataFrame,
        line_table: pd.DataFrame,
    ) -> dict[str, float | int]:
        """Construct the best parameter mapping."""

        best_global = global_table.iloc[0]

        best_parameters = (
            self.spamms_fit.parameters.current_values()
        )

        for name in GLOBAL_PARAMETERS:
            best_parameters[name] = self._scalar(
                best_global[name]
            )

        if self.sigma_mode == "shared":
            for name in SIGMA_PARAMETERS:
                best_parameters[name] = (
                    self._scalar(
                        best_global[name]
                    )
                )

        else:
            best_lines = line_table[
                line_table["global_rank"] == 1
            ]

            for row in best_lines.itertuples(
                index=False
            ):
                best_parameters[
                    f"{row.line_name}_sigma_R"
                ] = self._scalar(
                    row.sigma_R
                )

                best_parameters[
                    f"{row.line_name}_sigma_T"
                ] = self._scalar(
                    row.sigma_T
                )

            # Remove the template sigma values because they do not
            # describe a per-line solution.
            best_parameters.pop(
                "sigma_R",
                None,
            )

            best_parameters.pop(
                "sigma_T",
                None,
            )

        return best_parameters

    def _build_best_evaluation(
        self,
        best_parameters: dict[str, float | int],
        global_table: pd.DataFrame,
        line_table: pd.DataFrame,
    ) -> EvaluationDetails:
        """
        Load and prepare detailed arrays for the best grid solution.
        """

        best_line_rows = line_table[
            line_table["global_rank"] == 1
        ]

        expected_lines = set(
            self.spamms_fit.spectrum.line_names
        )

        line_counts = (
            best_line_rows["line_name"]
            .value_counts()
        )

        missing_lines = (
            expected_lines
            - set(line_counts.index)
        )

        duplicate_lines = {
            line_name: int(count)
            for line_name, count
            in line_counts.items()
            if count != 1
        }

        if missing_lines or duplicate_lines:
            raise RuntimeError(
                "The best grid solution must contain "
                "exactly one record for every selected line. "
                f"Missing lines: {sorted(missing_lines)}; "
                f"invalid line counts: {duplicate_lines}."
            )

        models: ModelSpectra = {}

        observed_wavelengths: dict[
            str,
            np.ndarray,
        ] = {}

        observed_fluxes: dict[
            str,
            np.ndarray,
        ] = {}

        observed_uncertainties: dict[
            str,
            np.ndarray,
        ] = {}

        interpolated_models: dict[
            str,
            np.ndarray,
        ] = {}

        residual_arrays: dict[
            str,
            np.ndarray,
        ] = {}

        chi2_by_line: dict[
            str,
            float,
        ] = {}

        total_normalization = 0.0

        for row in best_line_rows.itertuples(
            index=False
        ):
            line_name = row.line_name

            profile_path = (
                self._resolve_profile_path(
                    row.profile_path
                )
            )

            model_wave, model_flux = (
                self._read_model_profile(
                    profile_path
                )
            )

            observed = (
                self.spamms_fit.spectrum.get_line(
                    line_name
                )
            )

            interpolated_flux = (
                interpolate_model(
                    model_wavelength=model_wave,
                    model_flux=model_flux,
                    observed_wavelength=(
                        observed.wavelength
                    ),
                    extrapolate=(
                        self.spamms_fit.extrapolate
                    ),
                )
            )

            line_residuals = (
                observed.flux
                - interpolated_flux
            )

            line_chi2 = chi_square(
                observed_flux=observed.flux,
                model_flux=interpolated_flux,
                uncertainty=(
                    observed.uncertainty
                ),
            )

            # Retain the observed arrays for result plotting and
            # diagnostics.
            observed_wavelengths[line_name] = (
                observed.wavelength
            )

            observed_fluxes[line_name] = (
                observed.flux
            )

            observed_uncertainties[line_name] = (
                observed.uncertainty
            )

            # Retain the native and interpolated best model.
            models[line_name] = (
                model_wave,
                model_flux,
            )

            interpolated_models[line_name] = (
                interpolated_flux
            )

            residual_arrays[line_name] = (
                line_residuals
            )

            chi2_by_line[line_name] = (
                line_chi2
            )

            total_normalization += float(
                np.sum(
                    np.log(2.0 * np.pi)
                    + 2.0
                    * np.log(
                        observed.uncertainty
                    )
                )
            )

        total_chi2 = float(
            sum(chi2_by_line.values())
        )

        expected_chi2 = float(
            global_table.loc[
                0,
                "chi2_total",
            ]
        )

        if not np.isclose(
            total_chi2,
            expected_chi2,
        ):
            raise RuntimeError(
                "Recalculated best chi-square does not "
                "match the ranked global result. "
                f"Ranked value: {expected_chi2:.12g}; "
                f"recalculated value: "
                f"{total_chi2:.12g}."
            )

        reduced = reduced_chi_square(
            chi2=total_chi2,
            n_pixels=(
                self.spamms_fit.n_fitting_pixels
            ),
            n_free_parameters=(
                self._effective_n_free()
            ),
        )

        log_likelihood = float(
            -0.5
            * (
                total_chi2
                + total_normalization
            )
        )

        return {
            "parameters": best_parameters,
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

    def _effective_n_free(self) -> int:
        """
        Return the effective number of fitted parameters.

        In per-line mode, every free sigma parameter is repeated once
        for every selected line.
        """

        free_names = (
            self.spamms_fit.parameters.free_names()
        )

        global_free = sum(
            name in GLOBAL_PARAMETERS
            for name in free_names
        )

        sigma_free = sum(
            name in SIGMA_PARAMETERS
            for name in free_names
        )

        if self.sigma_mode == "shared":
            return global_free + sigma_free

        return (
            global_free
            + sigma_free
            * self.spamms_fit.spectrum.n_lines
        )

    def _result_free_parameter_names(
        self,
    ) -> tuple[str, ...]:
        """Return result parameter names, including per-line sigmas."""

        free_names = (
            self.spamms_fit.parameters.free_names()
        )

        if self.sigma_mode == "shared":
            return free_names

        result_names: list[str] = []

        for name in free_names:
            if name not in SIGMA_PARAMETERS:
                result_names.append(name)
                continue

            for line_name in (
                self.spamms_fit.spectrum.line_names
            ):
                result_names.append(
                    f"{line_name}_{name}"
                )

        return tuple(result_names)

    @staticmethod
    def _scalar(
        value: Any,
    ) -> float | int:
        """Convert NumPy and pandas scalars to Python scalars."""

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, (int, float)):
            return value

        return value

    @property
    def n_profile_evaluations(self) -> int:
        """Return the number of profile comparisons."""

        return self._n_profile_evaluations

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"ChiSquareFit("
            f"model_index="
            f"{str(self.model_index)!r}, "
            f"sigma_mode={self.sigma_mode!r}, "
            f"status={self.status!r}"
            f")"
        )
