"""Plotting for Bayesian SPAMMSFit results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from spammsfit.results.bayesian import (
    BayesianResult,
)


DEFAULT_PARAMETER_LABELS = {
    "teff": r"$T_{\rm eff}$",
    "r_pole": r"$R_{\rm pole}$",
    "mass": r"$M$",
    "inclination": r"$i$ (deg)",
    "v_crit_frac": r"$v/v_{\rm crit}$",
    "sigma_R": (
        r"$\sigma_R$ (km s$^{-1}$)"
    ),
    "sigma_T": (
        r"$\sigma_T$ (km s$^{-1}$)"
    ),
}


def plot_trace(
    result: BayesianResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    labels: Mapping[
        str,
        str,
    ] | None = None,
    burnin_marker: bool = True,
    figsize: tuple[
        float,
        float,
    ] | None = None,
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot walker traces for selected Bayesian parameters.
    """

    _validate_result(result)

    selected_parameters = (
        _prepare_parameters(
            result=result,
            parameters=parameters,
        )
    )

    parameter_labels = _prepare_labels(
        parameters=selected_parameters,
        labels=labels,
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
            10.0,
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

    parameter_indices = {
        name: index
        for index, name in enumerate(
            result.free_parameter_names
        )
    }

    burnin = int(
        result.settings.get(
            "burnin",
            0,
        )
    )

    for axis, parameter in zip(
        axes,
        selected_parameters,
        strict=True,
    ):
        parameter_index = (
            parameter_indices[
                parameter
            ]
        )

        axis.plot(
            result.chain[
                :,
                :,
                parameter_index,
            ],
            color="black",
            alpha=0.22,
            linewidth=0.6,
        )

        if burnin_marker and burnin > 0:
            axis.axvline(
                burnin,
                color="tab:blue",
                linewidth=1.2,
                linestyle=":",
                label="Burn-in",
            )

        if parameter in true_values:
            axis.axhline(
                true_values[parameter],
                color="tab:red",
                linewidth=1.2,
                linestyle="--",
                label="True value",
            )

        axis.set_ylabel(
            parameter_labels[
                parameter
            ],
            fontsize=11,
        )

        if (
            burnin_marker
            or parameter in true_values
        ):
            handles, legend_labels = (
                axis.get_legend_handles_labels()
            )

            if handles:
                axis.legend(
                    fontsize=9,
                    loc="best",
                )

        _style_axis(axis)

    axes[-1].set_xlabel(
        "Step",
        fontsize=13,
    )

    figure.subplots_adjust(
        hspace=0.08,
    )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def plot_corner(
    result: BayesianResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    labels: Mapping[
        str,
        str,
    ] | None = None,
    quantiles: tuple[
        float,
        float,
        float,
    ] = (
        0.16,
        0.50,
        0.84,
    ),
    show_titles: bool = True,
    bins: int = 20,
    color: str = "tab:blue",
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
    **corner_kwargs,
) -> Figure:
    """
    Create a posterior corner plot.
    """

    _validate_result(result)

    selected_parameters = (
        _prepare_parameters(
            result=result,
            parameters=parameters,
        )
    )

    parameter_indices = [
        result.free_parameter_names.index(
            parameter
        )
        for parameter in selected_parameters
    ]

    samples = result.posterior_samples[
        :,
        parameter_indices,
    ]

    parameter_labels = _prepare_labels(
        parameters=selected_parameters,
        labels=labels,
    )

    plot_labels = [
        parameter_labels[parameter]
        for parameter in selected_parameters
    ]

    if true_values is None:
        truths = None
    else:
        truths = [
            true_values.get(
                parameter,
                None,
            )
            for parameter
            in selected_parameters
        ]

    figure = corner.corner(
        samples,
        labels=plot_labels,
        truths=truths,
        quantiles=quantiles,
        show_titles=show_titles,
        bins=int(bins),
        color=color,
        **corner_kwargs,
    )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure


def plot_acceptance_fraction(
    result: BayesianResult,
    *,
    recommended_range: tuple[
        float,
        float,
    ] = (
        0.2,
        0.5,
    ),
    figsize: tuple[
        float,
        float,
    ] = (
        8.0,
        4.5,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """
    Plot acceptance fraction for every walker.
    """

    _validate_result(result)

    lower, upper = (
        recommended_range
    )

    lower = float(lower)
    upper = float(upper)

    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "recommended_range must satisfy "
            "0 <= lower < upper <= 1."
        )

    walker_numbers = np.arange(
        result.n_walkers
    )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    axis.axhspan(
        lower,
        upper,
        color="tab:green",
        alpha=0.15,
        label="Reference range",
    )

    axis.bar(
        walker_numbers,
        result.acceptance_fraction,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )

    axis.axhline(
        result.mean_acceptance_fraction,
        color="tab:red",
        linewidth=1.3,
        linestyle="--",
        label=(
            "Mean = "
            f"{result.mean_acceptance_fraction:.3f}"
        ),
    )

    axis.set_xlabel(
        "Walker",
        fontsize=13,
    )

    axis.set_ylabel(
        "Acceptance fraction",
        fontsize=13,
    )

    axis.set_ylim(
        0.0,
        1.0,
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


def plot_autocorrelation_time(
    result: BayesianResult,
    *,
    labels: Mapping[
        str,
        str,
    ] | None = None,
    figsize: tuple[
        float,
        float,
    ] = (
        8.0,
        4.8,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """
    Plot estimated integrated autocorrelation times.
    """

    _validate_result(result)

    if result.autocorrelation_time is None:
        raise ValueError(
            "Reliable autocorrelation-time estimates "
            "are not available for this result."
        )

    parameter_labels = _prepare_labels(
        parameters=(
            result.free_parameter_names
        ),
        labels=labels,
    )

    x_positions = np.arange(
        result.n_parameters
    )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    axis.bar(
        x_positions,
        result.autocorrelation_time,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            parameter_labels[name]
            for name in (
                result.free_parameter_names
            )
        ],
        rotation=30,
        ha="right",
    )

    axis.set_ylabel(
        "Autocorrelation time (steps)",
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


def plot_posterior_histograms(
    result: BayesianResult,
    *,
    parameters: Iterable[str] | None = None,
    true_values: Mapping[
        str,
        float,
    ] | None = None,
    labels: Mapping[
        str,
        str,
    ] | None = None,
    bins: int = 30,
    figsize: tuple[
        float,
        float,
    ] | None = None,
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot one-dimensional marginalized posterior distributions.
    """

    _validate_result(result)

    selected_parameters = (
        _prepare_parameters(
            result=result,
            parameters=parameters,
        )
    )

    parameter_labels = _prepare_labels(
        parameters=selected_parameters,
        labels=labels,
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
        squeeze=False,
    )

    axes = axes[:, 0]

    for axis, parameter in zip(
        axes,
        selected_parameters,
        strict=True,
    ):
        parameter_index = (
            result.free_parameter_names.index(
                parameter
            )
        )

        samples = result.posterior_samples[
            :,
            parameter_index,
        ]

        interval = result.interval(
            parameter
        )

        axis.hist(
            samples,
            bins=int(bins),
            density=True,
            color="tab:blue",
            alpha=0.70,
            edgecolor="black",
            linewidth=0.6,
        )

        axis.axvline(
            interval["p50"],
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="Posterior median",
        )

        axis.axvspan(
            interval["p16"],
            interval["p84"],
            color="tab:blue",
            alpha=0.15,
            label="16th–84th percentile",
        )

        if parameter in true_values:
            axis.axvline(
                true_values[parameter],
                color="tab:red",
                linewidth=1.2,
                linestyle=":",
                label="True value",
            )

        axis.set_xlabel(
            parameter_labels[
                parameter
            ],
            fontsize=12,
        )

        axis.set_ylabel(
            "Posterior density",
            fontsize=11,
        )

        axis.legend(
            fontsize=9,
        )

        _style_axis(axis)

    figure.subplots_adjust(
        hspace=0.35,
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
    result: BayesianResult,
    parameters: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate selected posterior parameters."""

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
            f"Unknown Bayesian parameters: "
            f"{sorted(unknown)}."
        )

    return selected


def _prepare_labels(
    parameters: Iterable[str],
    labels: Mapping[
        str,
        str,
    ] | None,
) -> dict[str, str]:
    """Prepare default and user-supplied plot labels."""

    prepared_labels = dict(
        DEFAULT_PARAMETER_LABELS
    )

    if labels is not None:
        prepared_labels.update(
            dict(labels)
        )

    return {
        parameter: prepared_labels.get(
            parameter,
            parameter,
        )
        for parameter in parameters
    }


def _validate_result(
    result: BayesianResult,
) -> None:
    """Validate the supplied result type."""

    if not isinstance(
        result,
        BayesianResult,
    ):
        raise TypeError(
            "result must be a "
            "BayesianResult instance."
        )


def _style_axis(
    axis: plt.Axes,
) -> None:
    """Apply shared Bayesian plot styling."""

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
