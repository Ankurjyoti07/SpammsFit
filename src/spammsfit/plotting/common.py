"""Plotting shared by all SPAMMSFit result types."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spammsfit.results.base import BaseResult


def plot_fit(
    result: BaseResult,
    *,
    lines: Iterable[str] | None = None,
    show_native_model: bool = False,
    figsize: tuple[float, float] | None = None,
    observed_color: str = "black",
    model_color: str = "tab:red",
    native_model_color: str = "tab:blue",
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot observed and best-fitting spectra.

    Parameters
    ----------
    result
        Completed SPAMMSFit result.
    lines
        Lines to plot. By default, all fitted lines are shown.
    show_native_model
        Also show the model on its native SPAMMS wavelength grid.
    figsize
        Figure width and height. A suitable value is chosen
        automatically when omitted.
    observed_color
        Colour of the observed spectrum.
    model_color
        Colour of the interpolated best-fitting model.
    native_model_color
        Colour of the native-grid SPAMMS model.
    save
        Optional output filename.
    dpi
        Resolution used when saving.
    show
        Display the figure with ``plt.show()``.

    Returns
    -------
    figure, axes
        Matplotlib figure and one-dimensional axes array.
    """

    selected_lines = _prepare_lines(
        result=result,
        lines=lines,
    )

    n_lines = len(selected_lines)

    if figsize is None:
        figsize = (
            9.0,
            max(3.2 * n_lines, 4.0),
        )

    figure, axes = plt.subplots(
        n_lines,
        1,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes[:, 0]

    for axis, line_name in zip(
        axes,
        selected_lines,
        strict=True,
    ):
        observed_wavelength = (
            result.evaluation[
                "observed_wavelengths"
            ][line_name]
        )

        observed_flux = (
            result.evaluation[
                "observed_fluxes"
            ][line_name]
        )

        interpolated_flux = (
            result.interpolated_models[
                line_name
            ]
        )

        axis.plot(
            observed_wavelength,
            observed_flux,
            color=observed_color,
            linewidth=0.9,
            alpha=0.65,
            label="Observed",
        )

        axis.plot(
            observed_wavelength,
            interpolated_flux,
            color=model_color,
            linewidth=1.8,
            label="Best fit",
        )

        if show_native_model:
            native_wavelength, native_flux = (
                result.models[line_name]
            )

            axis.plot(
                native_wavelength,
                native_flux,
                color=native_model_color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label="Native SPAMMS model",
            )

        line_chi2 = result.chi2_by_line[
            line_name
        ]

        axis.text(
            0.02,
            0.06,
            (
                f"{line_name}\n"
                rf"$\chi^2={line_chi2:.2f}$"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
        )

        axis.set_ylabel(
            "Normalized flux",
            fontsize=13,
        )

        axis.legend(
            loc="best",
            fontsize=10,
        )

        _style_axis(axis)

    axes[-1].set_xlabel(
        r"Wavelength ($\AA$)",
        fontsize=13,
    )

    figure.subplots_adjust(
        hspace=0.12,
    )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def plot_residuals(
    result: BaseResult,
    *,
    lines: Iterable[str] | None = None,
    show_uncertainty: bool = True,
    figsize: tuple[float, float] | None = None,
    residual_color: str = "tab:red",
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot observed-minus-model residuals.

    Parameters
    ----------
    result
        Completed SPAMMSFit result.
    lines
        Lines to plot. By default, all fitted lines are shown.
    show_uncertainty
        Show positive and negative one-sigma uncertainties.
    figsize
        Optional figure size.
    residual_color
        Colour of the residual curve.
    save
        Optional output filename.
    dpi
        Resolution used when saving.
    show
        Display the figure.

    Returns
    -------
    figure, axes
        Matplotlib figure and axes array.
    """

    selected_lines = _prepare_lines(
        result=result,
        lines=lines,
    )

    n_lines = len(selected_lines)

    if figsize is None:
        figsize = (
            9.0,
            max(2.7 * n_lines, 3.5),
        )

    figure, axes = plt.subplots(
        n_lines,
        1,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes[:, 0]

    for axis, line_name in zip(
        axes,
        selected_lines,
        strict=True,
    ):
        wavelength = result.evaluation[
            "observed_wavelengths"
        ][line_name]

        residual = result.residual_arrays[
            line_name
        ]

        uncertainty = result.evaluation[
            "observed_uncertainties"
        ][line_name]

        axis.plot(
            wavelength,
            residual,
            color=residual_color,
            linewidth=1.0,
        )

        axis.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle=":",
        )

        if show_uncertainty:
            axis.plot(
                wavelength,
                uncertainty,
                color="0.45",
                linewidth=0.9,
                linestyle="--",
                label=r"$\pm1\sigma$",
            )

            axis.plot(
                wavelength,
                -uncertainty,
                color="0.45",
                linewidth=0.9,
                linestyle="--",
            )

        rms = float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        )

        axis.text(
            0.02,
            0.08,
            (
                f"{line_name}\n"
                f"RMS = {rms:.4g}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
        )

        axis.set_ylabel(
            "Residual",
            fontsize=13,
        )

        if show_uncertainty:
            axis.legend(
                loc="best",
                fontsize=10,
            )

        _style_axis(axis)

    axes[-1].set_xlabel(
        r"Wavelength ($\AA$)",
        fontsize=13,
    )

    figure.subplots_adjust(
        hspace=0.12,
    )

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def plot_fit_with_residuals(
    result: BaseResult,
    *,
    line: str,
    show_uncertainty: bool = True,
    figsize: tuple[float, float] = (
        9.0,
        6.0,
    ),
    save: str | Path | None = None,
    dpi: int = 200,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Plot one fitted line and its residuals in two aligned panels.
    """

    selected_line = _prepare_lines(
        result=result,
        lines=[line],
    )[0]

    wavelength = result.evaluation[
        "observed_wavelengths"
    ][selected_line]

    observed_flux = result.evaluation[
        "observed_fluxes"
    ][selected_line]

    uncertainty = result.evaluation[
        "observed_uncertainties"
    ][selected_line]

    model_flux = result.interpolated_models[
        selected_line
    ]

    residual = result.residual_arrays[
        selected_line
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.05,
        },
    )

    axes[0].plot(
        wavelength,
        observed_flux,
        color="black",
        linewidth=0.9,
        alpha=0.65,
        label="Observed",
    )

    axes[0].plot(
        wavelength,
        model_flux,
        color="tab:red",
        linewidth=1.8,
        label="Best fit",
    )

    axes[0].set_ylabel(
        "Normalized flux",
        fontsize=13,
    )

    axes[0].legend(
        fontsize=10,
    )

    axes[0].text(
        0.02,
        0.06,
        selected_line,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )

    axes[1].plot(
        wavelength,
        residual,
        color="tab:red",
        linewidth=1.0,
    )

    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
    )

    if show_uncertainty:
        axes[1].plot(
            wavelength,
            uncertainty,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
        )

        axes[1].plot(
            wavelength,
            -uncertainty,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
        )

    axes[1].set_xlabel(
        r"Wavelength ($\AA$)",
        fontsize=13,
    )

    axes[1].set_ylabel(
        "Residual",
        fontsize=11,
    )

    for axis in axes:
        _style_axis(axis)

    _save_figure(
        figure=figure,
        path=save,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure, axes


def _prepare_lines(
    result: BaseResult,
    lines: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate requested result line names."""

    available = result.line_names

    if lines is None:
        return available

    if isinstance(lines, str):
        lines = [lines]

    selected = tuple(lines)

    if not selected:
        raise ValueError(
            "At least one line must be selected."
        )

    unknown = (
        set(selected)
        - set(available)
    )

    if unknown:
        raise KeyError(
            f"Unknown result lines: "
            f"{sorted(unknown)}. "
            f"Available lines: {available}."
        )

    return selected


def _style_axis(
    axis: Axes,
) -> None:
    """Apply shared plot styling."""

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=11,
        length=4,
    )


def _save_figure(
    figure: Figure,
    path: str | Path | None,
    dpi: int,
) -> None:
    """Save a figure when an output path is supplied."""

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
