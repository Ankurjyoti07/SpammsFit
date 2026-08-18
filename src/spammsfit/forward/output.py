"""Reading and validation of SPAMMS model spectra."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

# One model line is represented by:
# (wavelength array, flux array)
ModelLine = tuple[FloatArray, FloatArray]

# A multiline SPAMMS calculation is represented by:
# {"HEII4200": (wavelength, flux), ...}
ModelSpectra = dict[str, ModelLine]


def read_model_spectra(
    output_directory: str | Path,
    selected_lines: Iterable[str],
) -> ModelSpectra:
    """
    Read all requested line profiles from one SPAMMS calculation.

    Parameters
    ----------
    output_directory
        Root output directory used by the SPAMMS calculation.
    selected_lines
        Names of the lines expected in the output.

    Returns
    -------
    dict
        Mapping from each line name to its wavelength and flux arrays.

    Notes
    -----
    The output directory is scanned only once, even for a multiline
    calculation.
    """

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    if not output_directory.is_dir():
        raise FileNotFoundError(
            "SPAMMS output directory does not exist: "
            f"{output_directory}"
        )

    lines = _prepare_line_names(
        selected_lines,
    )

    candidate_files = tuple(
        output_directory.rglob("hjd*.txt")
    )

    if not candidate_files:
        raise FileNotFoundError(
            "No SPAMMS model files matching 'hjd*.txt' "
            f"were found in {output_directory}."
        )

    matched_files = _match_model_files(
        candidate_files=candidate_files,
        selected_lines=lines,
        output_directory=output_directory,
    )

    models: ModelSpectra = {}

    for line_name in lines:
        wavelength, flux = _read_model_file(
            matched_files[line_name],
            line_name=line_name,
        )

        models[line_name] = (
            wavelength,
            flux,
        )

    return models


def _match_model_files(
    candidate_files: Iterable[Path],
    selected_lines: tuple[str, ...],
    output_directory: Path,
) -> dict[str, Path]:
    """
    Match exactly one model file to each requested line name.
    """

    matches: dict[str, list[Path]] = {
        line_name: []
        for line_name in selected_lines
    }

    for model_file in candidate_files:
        filename = model_file.name

        for line_name in selected_lines:
            expected_suffix = (
                f"_{line_name}.txt"
            )

            if filename.endswith(expected_suffix):
                matches[line_name].append(
                    model_file
                )

    missing = [
        line_name
        for line_name, files in matches.items()
        if not files
    ]

    if missing:
        raise FileNotFoundError(
            "No SPAMMS output file was found for "
            f"line(s) {missing} in {output_directory}."
        )

    duplicates = {
        line_name: files
        for line_name, files in matches.items()
        if len(files) > 1
    }

    if duplicates:
        details = "; ".join(
            (
                f"{line_name}: "
                + ", ".join(
                    str(path)
                    for path in files
                )
            )
            for line_name, files
            in duplicates.items()
        )

        raise RuntimeError(
            "Expected exactly one SPAMMS model file "
            f"per line, but found duplicates: {details}"
        )

    return {
        line_name: files[0]
        for line_name, files in matches.items()
    }


def _read_model_file(
    model_file: Path,
    line_name: str,
) -> ModelLine:
    """
    Read and validate one SPAMMS line-profile file.
    """

    try:
        model = np.loadtxt(
            model_file,
            dtype=np.float64,
            usecols=(0, 1),
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not read SPAMMS model for "
            f"{line_name!r}: {model_file}"
        ) from error

    model = np.atleast_2d(model)

    if model.shape[1] != 2:
        raise ValueError(
            f"SPAMMS model {model_file} does not contain "
            "readable wavelength and flux columns."
        )

    if model.shape[0] < 2:
        raise ValueError(
            f"SPAMMS model {model_file} contains fewer "
            "than two wavelength pixels."
        )

    wavelength = np.ascontiguousarray(
        model[:, 0],
        dtype=np.float64,
    )

    flux = np.ascontiguousarray(
        model[:, 1],
        dtype=np.float64,
    )

    _validate_model_arrays(
        wavelength=wavelength,
        flux=flux,
        line_name=line_name,
        model_file=model_file,
    )

    # The arrays represent a completed forward model and should not
    # be modified accidentally by later likelihood calculations.
    wavelength.flags.writeable = False
    flux.flags.writeable = False

    return wavelength, flux


def _validate_model_arrays(
    wavelength: FloatArray,
    flux: FloatArray,
    line_name: str,
    model_file: Path,
) -> None:
    """Validate one model wavelength/flux pair."""

    if wavelength.ndim != 1:
        raise ValueError(
            f"Wavelength array for {line_name!r} "
            f"in {model_file} is not one-dimensional."
        )

    if flux.ndim != 1:
        raise ValueError(
            f"Flux array for {line_name!r} "
            f"in {model_file} is not one-dimensional."
        )

    if wavelength.shape != flux.shape:
        raise ValueError(
            f"Wavelength and flux shapes differ for "
            f"{line_name!r} in {model_file}."
        )

    if not np.all(np.isfinite(wavelength)):
        raise ValueError(
            f"Model wavelengths for {line_name!r} "
            "contain non-finite values."
        )

    if not np.all(np.isfinite(flux)):
        raise ValueError(
            f"Model fluxes for {line_name!r} "
            "contain non-finite values."
        )

    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError(
            f"Model wavelengths for {line_name!r} "
            "must be strictly increasing."
        )


def _prepare_line_names(
    selected_lines: Iterable[str],
) -> tuple[str, ...]:
    """Validate requested line names."""

    if isinstance(selected_lines, str):
        raise TypeError(
            "selected_lines must be an iterable of line "
            "names, not a single string."
        )

    prepared: list[str] = []

    for line_name in selected_lines:
        if not isinstance(line_name, str):
            raise TypeError(
                "Every selected line name must be a string."
            )

        line_name = line_name.strip()

        if not line_name:
            raise ValueError(
                "Selected line names cannot be empty."
            )

        prepared.append(line_name)

    if not prepared:
        raise ValueError(
            "At least one model line must be requested."
        )

    if len(set(prepared)) != len(prepared):
        raise ValueError(
            "selected_lines contains duplicate names."
        )

    return tuple(prepared)
