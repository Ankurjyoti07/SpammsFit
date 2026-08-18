"""Plotting for grid-based chi-square results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from spammsfit.results.chisquare import (
    ChiSquareResult,
)


def plot_profile_likelihood(
    result: ChiSquareResult,
    parameter: str,
    *,
    true_value: float | None = None,
    best_color: str = "black",
    true_color: str = "tab:red",
    figsize: tuple[float, float] = (
        7.0,
        5.0,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """
    Plot profile delta-chi-square for one grid parameter.
    """

    _validate_result(result)

    profile = result.profile_likelihood(
        parameter
    )

    best_value = result.best_global_row[
        parameter
    ]

    figure, axis = plt.subplots(
        figsize=figsize
    )

    axis.plot(
        profile[parameter],
        profile["delta_chi2"],
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )

    axis.axvline(
        best_value,
        color=best_color,
        linewidth=1.3,
        linestyle="--",
        label="Best fit",
    )

    if true_value is not None:
        axis.axvline(
            true_value,
            color=true_color,
            linewidth=1.3,
            linestyle=":",
            label="True value",
        )

    axis.set_xlabel(
        parameter,
        fontsize=13,
    )

    axis.set_ylabel(
        r"$\Delta\chi^2$",
        fontsize=13,
    )

    axis.legend(
        fontsize=10,
    )

    _style_axis(axis)

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axis


def plot_parameter_map(
    result: ChiSquareResult,
    x_parameter: str,
    y_parameter: str,
    *,
    true_values: tuple[
        float,
        float,
    ] | None = None,
    color_percentile: float = 95.0,
    delta_chi2_max: float | None = None,
    cmap: str = "viridis_r",
    figsize: tuple[float, float] = (
        7.0,
        6.0,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """
    Plot a profiled two-parameter delta-chi-square map.
    """

    _validate_result(result)

    parameter_map = result.parameter_map(
        x_parameter=x_parameter,
        y_parameter=y_parameter,
    )

    color_limit = _calculate_color_limit(
        delta_chi2=parameter_map[
            "delta_chi2"
        ].to_numpy(
            dtype=float
        ),
        color_percentile=color_percentile,
        delta_chi2_max=delta_chi2_max,
    )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    scatter = axis.scatter(
        parameter_map[x_parameter],
        parameter_map[y_parameter],
        c=parameter_map["delta_chi2"],
        cmap=cmap,
        vmin=0.0,
        vmax=color_limit,
        s=48,
        edgecolors="none",
    )

    best_x = result.best_global_row[
        x_parameter
    ]

    best_y = result.best_global_row[
        y_parameter
    ]

    axis.scatter(
        best_x,
        best_y,
        marker="X",
        s=150,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        label="Best fit",
        zorder=5,
    )

    if true_values is not None:
        axis.scatter(
            true_values[0],
            true_values[1],
            marker="+",
            s=180,
            color="tab:red",
            linewidth=2.0,
            label="True value",
            zorder=6,
        )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
    )

    colorbar.set_label(
        r"$\Delta\chi^2$",
        fontsize=12,
    )

    colorbar.ax.tick_params(
        labelsize=10,
    )

    axis.set_xlabel(
        x_parameter,
        fontsize=13,
    )

    axis.set_ylabel(
        y_parameter,
        fontsize=13,
    )

    axis.legend(
        fontsize=10,
    )

    _style_axis(axis)

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axis


def plot_chi2_corner(
    result: ChiSquareResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    color_percentile: float = 95.0,
    delta_chi2_max: float | None = None,
    cmap: str = "viridis_r",
    figsize: tuple[
        float,
        float,
    ] | None = None,
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Create a corner-style visualization of the chi-square landscape.

    Diagonal panels show one-dimensional profile delta-chi-square.
    Lower-triangle panels show two-dimensional profiled landscapes.

    Notes
    -----
    This is not a Bayesian posterior corner plot. It visualizes
    profile chi-square values from the discrete model grid.
    """

    _validate_result(result)

    selected_parameters = (
        _prepare_corner_parameters(
            result=result,
            parameters=parameters,
        )
    )

    n_parameters = len(
        selected_parameters
    )

    if figsize is None:
        panel_size = 2.7

        figsize = (
            panel_size * n_parameters,
            panel_size * n_parameters,
        )

    true_values = (
        {}
        if true_values is None
        else dict(true_values)
    )

    global_delta_chi2 = result.global_table[
        "delta_chi2"
    ].to_numpy(
        dtype=float
    )

    color_limit = _calculate_color_limit(
        delta_chi2=global_delta_chi2,
        color_percentile=color_percentile,
        delta_chi2_max=delta_chi2_max,
    )

    figure, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=figsize,
        squeeze=False,
    )

    color_mappable = None

    for row_index, y_parameter in enumerate(
        selected_parameters
    ):
        for column_index, x_parameter in enumerate(
            selected_parameters
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            if column_index > row_index:
                axis.set_visible(False)
                continue

            if row_index == column_index:
                profile = (
                    result.profile_likelihood(
                        x_parameter
                    )
                )

                axis.plot(
                    profile[x_parameter],
                    profile["delta_chi2"],
                    color="tab:blue",
                    linewidth=1.5,
                )

                best_value = (
                    result.best_global_row[
                        x_parameter
                    ]
                )

                axis.axvline(
                    best_value,
                    color="black",
                    linewidth=1.1,
                    linestyle="--",
                )

                if x_parameter in true_values:
                    axis.axvline(
                        true_values[
                            x_parameter
                        ],
                        color="tab:red",
                        linewidth=1.1,
                        linestyle=":",
                    )

                axis.set_ylim(
                    bottom=0.0
                )

                axis.set_ylabel(
                    r"$\Delta\chi^2$",
                    fontsize=10,
                )

            else:
                parameter_map = (
                    result.parameter_map(
                        x_parameter,
                        y_parameter,
                    )
                )

                color_mappable = axis.scatter(
                    parameter_map[
                        x_parameter
                    ],
                    parameter_map[
                        y_parameter
                    ],
                    c=parameter_map[
                        "delta_chi2"
                    ],
                    cmap=cmap,
                    vmin=0.0,
                    vmax=color_limit,
                    s=24,
                    edgecolors="none",
                )

                axis.scatter(
                    result.best_global_row[
                        x_parameter
                    ],
                    result.best_global_row[
                        y_parameter
                    ],
                    marker="X",
                    s=85,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=5,
                )

                if (
                    x_parameter in true_values
                    and y_parameter in true_values
                ):
                    axis.scatter(
                        true_values[
                            x_parameter
                        ],
                        true_values[
                            y_parameter
                        ],
                        marker="+",
                        s=100,
                        color="tab:red",
                        linewidth=1.6,
                        zorder=6,
                    )

            if row_index == (
                n_parameters - 1
            ):
                axis.set_xlabel(
                    x_parameter,
                    fontsize=11,
                )
            else:
                axis.tick_params(
                    labelbottom=False
                )

            if (
                column_index == 0
                and row_index
                != column_index
            ):
                axis.set_ylabel(
                    y_parameter,
                    fontsize=11,
                )
            elif row_index != column_index:
                axis.tick_params(
                    labelleft=False
                )

            _style_axis(axis)

    figure.subplots_adjust(
        hspace=0.08,
        wspace=0.08,
        right=0.90,
    )

    if color_mappable is not None:
        visible_axes = [
            axis
            for row in axes
            for axis in row
            if axis.get_visible()
        ]

        colorbar = figure.colorbar(
            color_mappable,
            ax=visible_axes,
            fraction=0.025,
            pad=0.02,
        )

        colorbar.set_label(
            r"$\Delta\chi^2$",
            fontsize=12,
        )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def _prepare_corner_parameters(
    result: ChiSquareResult,
    parameters: Iterable[str] | None,
) -> tuple[str, ...]:
    """Select valid numeric grid parameters."""

    if parameters is None:
        preferred = (
            "teff",
            "r_pole",
            "mass",
            "inclination",
            "v_crit_frac",
            "sigma_R",
            "sigma_T",
        )

        selected = tuple(
            parameter
            for parameter in preferred
            if (
                parameter
                in result.global_table.columns
                and result.global_table[
                    parameter
                ].nunique()
                > 1
            )
        )

    else:
        if isinstance(parameters, str):
            parameters = [parameters]

        selected = tuple(parameters)

    if not selected:
        raise ValueError(
            "No varying parameters are available "
            "for the chi-square corner plot."
        )

    missing = [
        parameter
        for parameter in selected
        if parameter
        not in result.global_table.columns
    ]

    if missing:
        raise KeyError(
            "Parameters are not present in the "
            f"global result table: {missing}."
        )

    nonvarying = [
        parameter
        for parameter in selected
        if result.global_table[
            parameter
        ].nunique()
        < 2
    ]

    if nonvarying:
        raise ValueError(
            "Corner-plot parameters must have at "
            f"least two values: {nonvarying}."
        )

    return selected


def _calculate_color_limit(
    delta_chi2: np.ndarray,
    color_percentile: float,
    delta_chi2_max: float | None,
) -> float:
    """Determine a robust upper colour limit."""

    if delta_chi2_max is not None:
        color_limit = float(
            delta_chi2_max
        )

        if color_limit <= 0.0:
            raise ValueError(
                "delta_chi2_max must be positive."
            )

        return color_limit

    color_percentile = float(
        color_percentile
    )

    if not (
        0.0
        < color_percentile
        <= 100.0
    ):
        raise ValueError(
            "color_percentile must lie in "
            "(0, 100]."
        )

    color_limit = float(
        np.percentile(
            delta_chi2,
            color_percentile,
        )
    )

    if color_limit <= 0.0:
        positive = delta_chi2[
            delta_chi2 > 0.0
        ]

        if positive.size:
            color_limit = float(
                np.max(positive)
            )
        else:
            color_limit = 1.0

    return color_limit


def _validate_result(
    result: ChiSquareResult,
) -> None:
    """Validate the supplied result type."""

    if not isinstance(
        result,
        ChiSquareResult,
    ):
        raise TypeError(
            "result must be a "
            "ChiSquareResult instance."
        )


def _style_axis(
    axis: plt.Axes,
) -> None:
    """Apply shared chi-square plot styling."""

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=10,
        length=4,
    )


def _save_figure(
    figure: Figure,
    path: str | Path | None,
    dpi: int,
) -> None:
    """Save a figure when requested."""

    if path is None:
        return

    output_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=int(dpi),
        bbox_inches="tight",
    )
