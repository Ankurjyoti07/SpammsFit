"""Plotting for differential-evolution results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from spammsfit.results.differential_evolution import (
    DEResult,
)


def plot_convergence(
    result: DEResult,
    *,
    log_scale: bool = False,
    figsize: tuple[float, float] = (
        7.0,
        5.0,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """Plot the best chi-square as a function of DE generation."""

    _validate_result(result)

    history = result.convergence_table()

    if history.empty:
        raise ValueError(
            "The DE result contains no generation history."
        )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    axis.plot(
        history["generation"],
        history["chi2"],
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )

    if log_scale:
        if np.any(
            history["chi2"] <= 0.0
        ):
            raise ValueError(
                "A logarithmic chi-square axis requires "
                "strictly positive values."
            )

        axis.set_yscale("log")

    axis.set_xlabel(
        "Generation",
        fontsize=13,
    )

    axis.set_ylabel(
        r"Best $\chi^2$",
        fontsize=13,
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


def plot_parameter_evolution(
    result: DEResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    figsize: tuple[
        float,
        float,
    ] | None = None,
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot the best parameter values after each DE generation.
    """

    _validate_result(result)

    history = result.convergence_table()

    if history.empty:
        raise ValueError(
            "The DE result contains no generation history."
        )

    selected_parameters = (
        _prepare_parameters(
            result=result,
            parameters=parameters,
        )
    )

    true_values = (
        {}
        if true_values is None
        else dict(true_values)
    )

    n_parameters = len(
        selected_parameters
    )

    if figsize is None:
        figsize = (
            8.0,
            max(
                2.5 * n_parameters,
                4.0,
            ),
        )

    figure, axes = plt.subplots(
        n_parameters,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    for axis, parameter in zip(
        axes,
        selected_parameters,
        strict=True,
    ):
        axis.plot(
            history["generation"],
            history[parameter],
            color="tab:blue",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

        if parameter in true_values:
            axis.axhline(
                true_values[parameter],
                color="tab:red",
                linewidth=1.2,
                linestyle="--",
                label="True value",
            )

            axis.legend(
                fontsize=10,
            )

        axis.set_ylabel(
            parameter,
            fontsize=11,
        )

        _style_axis(axis)

    axes[-1].set_xlabel(
        "Generation",
        fontsize=13,
    )

    figure.subplots_adjust(
        hspace=0.10,
    )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def plot_population(
    result: DEResult,
    x_parameter: str,
    y_parameter: str,
    *,
    true_values: tuple[
        float,
        float,
    ] | None = None,
    color_percentile: float = 95.0,
    chi2_max: float | None = None,
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
    Plot the final DE population in two parameter dimensions.

    Notes
    -----
    The final population is an optimizer diagnostic, not a posterior
    distribution.
    """

    _validate_result(result)

    population = result.population_table()

    for parameter in (
        x_parameter,
        y_parameter,
    ):
        if parameter not in (
            result.free_parameter_names
        ):
            raise KeyError(
                f"Unknown DE parameter "
                f"{parameter!r}. Available parameters: "
                f"{result.free_parameter_names}."
            )

    color_limit = _calculate_color_limit(
        chi2_values=population[
            "chi2"
        ].to_numpy(
            dtype=float
        ),
        percentile=color_percentile,
        explicit_maximum=chi2_max,
    )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    scatter = axis.scatter(
        population[x_parameter],
        population[y_parameter],
        c=population["chi2"],
        cmap=cmap,
        vmin=float(
            population["chi2"].min()
        ),
        vmax=color_limit,
        s=55,
        edgecolor="black",
        linewidth=0.35,
    )

    best_parameters = {
        name: value
        for name, value in zip(
            result.free_parameter_names,
            result.best_vector,
            strict=True,
        )
    }

    axis.scatter(
        best_parameters[x_parameter],
        best_parameters[y_parameter],
        marker="X",
        s=160,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        label="DE best fit",
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
        r"$\chi^2$",
        fontsize=12,
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


def plot_population_corner(
    result: DEResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    color_percentile: float = 95.0,
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
    Plot the final DE population in a corner-style layout.

    Diagonal panels show final-population parameter distributions.
    Lower panels show parameter pairs coloured by chi-square.
    """

    _validate_result(result)

    selected_parameters = (
        _prepare_parameters(
            result=result,
            parameters=parameters,
        )
    )

    true_values = (
        {}
        if true_values is None
        else dict(true_values)
    )

    population = result.population_table()

    n_parameters = len(
        selected_parameters
    )

    if figsize is None:
        panel_size = 2.7

        figsize = (
            panel_size * n_parameters,
            panel_size * n_parameters,
        )

    color_limit = _calculate_color_limit(
        chi2_values=population[
            "chi2"
        ].to_numpy(
            dtype=float
        ),
        percentile=color_percentile,
        explicit_maximum=None,
    )

    chi2_minimum = float(
        population["chi2"].min()
    )

    best_values = {
        name: value
        for name, value in zip(
            result.free_parameter_names,
            result.best_vector,
            strict=True,
        )
    }

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
                axis.hist(
                    population[x_parameter],
                    bins="auto",
                    color="tab:blue",
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.6,
                )

                axis.axvline(
                    best_values[x_parameter],
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

            else:
                color_mappable = axis.scatter(
                    population[x_parameter],
                    population[y_parameter],
                    c=population["chi2"],
                    cmap=cmap,
                    vmin=chi2_minimum,
                    vmax=color_limit,
                    s=22,
                    edgecolors="none",
                )

                axis.scatter(
                    best_values[x_parameter],
                    best_values[y_parameter],
                    marker="X",
                    s=80,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
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
                        linewidth=1.5,
                        zorder=6,
                    )

            if row_index == (
                n_parameters - 1
            ):
                axis.set_xlabel(
                    x_parameter,
                    fontsize=10,
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
                    fontsize=10,
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
            r"$\chi^2$",
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


def _prepare_parameters(
    result: DEResult,
    parameters: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate selected free parameters."""

    if parameters is None:
        return result.free_parameter_names

    if isinstance(parameters, str):
        parameters = [parameters]

    selected = tuple(parameters)

    if not selected:
        raise ValueError(
            "At least one parameter must be selected."
        )

    unknown = (
        set(selected)
        - set(result.free_parameter_names)
    )

    if unknown:
        raise KeyError(
            f"Unknown DE parameters: "
            f"{sorted(unknown)}."
        )

    return selected


def _calculate_color_limit(
    chi2_values: np.ndarray,
    percentile: float,
    explicit_maximum: float | None,
) -> float:
    """Calculate a robust upper chi-square colour limit."""

    minimum = float(
        np.min(chi2_values)
    )

    if explicit_maximum is not None:
        maximum = float(
            explicit_maximum
        )

        if maximum <= minimum:
            raise ValueError(
                "chi2_max must exceed the minimum chi-square."
            )

        return maximum

    percentile = float(percentile)

    if not 0.0 < percentile <= 100.0:
        raise ValueError(
            "color_percentile must lie in (0, 100]."
        )

    maximum = float(
        np.percentile(
            chi2_values,
            percentile,
        )
    )

    if maximum <= minimum:
        maximum = float(
            np.max(chi2_values)
        )

    if maximum <= minimum:
        maximum = minimum + 1.0

    return maximum


def _validate_result(
    result: DEResult,
) -> None:
    """Validate the result type."""

    if not isinstance(
        result,
        DEResult,
    ):
        raise TypeError(
            "result must be a DEResult instance."
        )


def _style_axis(
    axis: plt.Axes,
) -> None:
    """Apply shared DE plot styling."""

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
