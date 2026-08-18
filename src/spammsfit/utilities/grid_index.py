"""Creation and inspection of SPAMMS grid-index files."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIRECTORY_PATTERN = re.compile(
    r"^Model_(\d+)$"
)

INDEX_PARAMETERS = (
    "teff",
    "r_pole",
    "mass",
    "inclination",
    "v_crit_frac",
    "sigma_R",
    "sigma_T",
)


def create_model_index(
    grid_directory: str | Path,
    *,
    input_file: str | Path | None = None,
    output_file: str | Path | None = None,
    relative_paths: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a model-index table for a precomputed SPAMMS grid.

    Parameters
    ----------
    grid_directory
        Directory containing ``Model_XXXX`` folders.
    input_file
        SPAMMS input file used to construct the grid. By default,
        ``grid_directory/input.txt`` is used.
    output_file
        Destination Parquet file. By default,
        ``grid_directory/model_index.parquet`` is used.
    relative_paths
        Store file paths relative to the model-index directory when
        True. Relative paths make a grid portable between systems.
    overwrite
        Replace an existing model-index file when True.
    verbose
        Print a summary after constructing the index.

    Returns
    -------
    pandas.DataFrame
        Complete model-index table.
    """

    grid_directory = (
        Path(grid_directory)
        .expanduser()
        .resolve()
    )

    if not grid_directory.is_dir():
        raise NotADirectoryError(
            f"Grid directory does not exist: "
            f"{grid_directory}"
        )

    if input_file is None:
        input_file = (
            grid_directory / "input.txt"
        )
    else:
        input_file = (
            Path(input_file)
            .expanduser()
            .resolve()
        )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Grid input file not found: "
            f"{input_file}"
        )

    if output_file is None:
        output_file = (
            grid_directory
            / "model_index.parquet"
        )
    else:
        output_file = (
            Path(output_file)
            .expanduser()
            .resolve()
        )

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Model index already exists: "
            f"{output_file}. Use overwrite=True "
            "to replace it."
        )

    input_parameters = parse_input_file(
        input_file
    )

    selected_lines = _get_selected_lines(
        input_parameters
    )

    times = _ensure_list(
        input_parameters.get(
            "times",
            [0.0],
        )
    )

    helium_abundances = _ensure_list(
        input_parameters.get(
            "he_abundances",
            [None],
        )
    )

    cno_abundances = _ensure_list(
        input_parameters.get(
            "cno_abundances",
            [None],
        )
    )

    model_directories = find_model_directories(
        grid_directory
    )

    if not model_directories:
        raise ValueError(
            f"No Model_XXXX directories were found "
            f"in {grid_directory}."
        )

    rows: list[dict[str, Any]] = []
    missing_model_info: list[str] = []

    index_directory = (
        output_file.parent.resolve()
    )

    for model_directory in model_directories:
        model_number = _model_number(
            model_directory
        )

        model_id = model_directory.name

        model_info_file = (
            model_directory
            / "model_info.txt"
        )

        if not model_info_file.is_file():
            missing_model_info.append(
                model_id
            )
            continue

        model_information = (
            parse_model_info(
                model_info_file
            )
        )

        for helium in helium_abundances:
            for cno in cno_abundances:
                abundance_directory = (
                    model_directory
                    / abundance_folder_name(
                        helium,
                        cno,
                    )
                )

                for time_value in times:
                    time_value = float(
                        time_value
                    )

                    for line_name in selected_lines:
                        profile_name = (
                            f"hjd{time_value:.11f}_"
                            f"{line_name}.txt"
                        )

                        profile_path = (
                            abundance_directory
                            / profile_name
                        )

                        rows.append(
                            {
                                "model_number": (
                                    model_number
                                ),
                                "model_id": model_id,
                                "model_dir": (
                                    _stored_path(
                                        model_directory,
                                        index_directory,
                                        relative_paths,
                                    )
                                ),
                                "model_info_path": (
                                    _stored_path(
                                        model_info_file,
                                        index_directory,
                                        relative_paths,
                                    )
                                ),
                                "abundance_dir": (
                                    _stored_path(
                                        abundance_directory,
                                        index_directory,
                                        relative_paths,
                                    )
                                ),
                                "he_abundance": (
                                    helium
                                ),
                                "cno_abundance": (
                                    cno
                                ),
                                "time": time_value,
                                "line_name": (
                                    line_name
                                ),
                                "profile_path": (
                                    _stored_path(
                                        profile_path,
                                        index_directory,
                                        relative_paths,
                                    )
                                ),
                                "profile_exists": (
                                    profile_path.is_file()
                                ),
                                "teff": (
                                    model_information.get(
                                        "teff",
                                        input_parameters.get(
                                            "teff"
                                        ),
                                    )
                                ),
                                "r_pole": (
                                    model_information.get(
                                        "r_pole",
                                        input_parameters.get(
                                            "r_pole"
                                        ),
                                    )
                                ),
                                "mass": (
                                    model_information.get(
                                        "mass",
                                        input_parameters.get(
                                            "mass"
                                        ),
                                    )
                                ),
                                "inclination": (
                                    model_information.get(
                                        "inclination"
                                    )
                                ),
                                "v_crit_frac": (
                                    model_information.get(
                                        "v_crit_frac"
                                    )
                                ),
                                "sigma_R": (
                                    model_information.get(
                                        "sigma_R"
                                    )
                                ),
                                "sigma_T": (
                                    model_information.get(
                                        "sigma_T"
                                    )
                                ),
                            }
                        )

    if not rows:
        raise RuntimeError(
            "No model-index records could be created."
        )

    model_index = pd.DataFrame(rows)

    validate_model_index(
        model_index
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        model_index.to_parquet(
            output_file,
            index=False,
        )
    except ImportError as error:
        raise ImportError(
            "Writing a Parquet model index requires "
            "either pyarrow or fastparquet."
        ) from error

    if verbose:
        print_model_index_summary(
            model_index=model_index,
            output_file=output_file,
            missing_model_info=(
                missing_model_info
            ),
        )

    return model_index


def parse_input_file(
    input_file: str | Path,
) -> dict[str, Any]:
    """
    Parse active assignments from a SPAMMS input file.
    """

    input_file = (
        Path(input_file)
        .expanduser()
        .resolve()
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    parameters: dict[str, Any] = {}

    for line_number, raw_line in enumerate(
        input_file.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, raw_value = line.split(
            "=",
            maxsplit=1,
        )

        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            raise ValueError(
                f"Empty parameter name on line "
                f"{line_number}."
            )

        if key in parameters:
            raise ValueError(
                f"Duplicate active parameter "
                f"{key!r} on line {line_number}."
            )

        parameters[key] = _parse_value(
            raw_value
        )

    return parameters


def parse_model_info(
    model_info_file: str | Path,
) -> dict[str, Any]:
    """
    Read numerical values from ``model_info.txt``.

    Expected entries have the form:

    ``sigma_R:100.0``
    """

    model_info_file = (
        Path(model_info_file)
        .expanduser()
        .resolve()
    )

    if not model_info_file.is_file():
        raise FileNotFoundError(
            f"model_info.txt not found: "
            f"{model_info_file}"
        )

    values: dict[str, Any] = {}

    for line_number, raw_line in enumerate(
        model_info_file.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        key, raw_value = line.split(
            ":",
            maxsplit=1,
        )

        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            raise ValueError(
                f"Empty model-info key on line "
                f"{line_number} in "
                f"{model_info_file}."
            )

        if key in values:
            raise ValueError(
                f"Duplicate model-info key {key!r} "
                f"in {model_info_file}."
            )

        values[key] = _parse_value(
            raw_value
        )

    return values


def find_model_directories(
    grid_directory: str | Path,
) -> tuple[Path, ...]:
    """
    Return valid Model_XXXX directories in numerical order.
    """

    grid_directory = (
        Path(grid_directory)
        .expanduser()
        .resolve()
    )

    model_directories = []

    for path in grid_directory.glob(
        "Model_*"
    ):
        if not path.is_dir():
            continue

        if MODEL_DIRECTORY_PATTERN.fullmatch(
            path.name
        ) is None:
            continue

        model_directories.append(path)

    return tuple(
        sorted(
            model_directories,
            key=_model_number,
        )
    )


def abundance_folder_name(
    helium: Any,
    cno: Any,
) -> str:
    """Return the SPAMMS abundance-directory name."""

    return f"He{helium}_CNO{cno}"


def validate_model_index(
    model_index: pd.DataFrame,
) -> None:
    """Validate a newly constructed model-index table."""

    required_columns = {
        "model_number",
        "model_id",
        "line_name",
        "profile_path",
        "profile_exists",
        *INDEX_PARAMETERS,
    }

    missing = (
        required_columns
        - set(model_index.columns)
    )

    if missing:
        raise ValueError(
            "Model index is missing required columns: "
            f"{sorted(missing)}."
        )

    if model_index.empty:
        raise ValueError(
            "Model index cannot be empty."
        )

    duplicate_columns = [
        "model_id",
        "he_abundance",
        "cno_abundance",
        "time",
        "line_name",
    ]

    duplicates = model_index.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = model_index.loc[
            duplicates,
            duplicate_columns,
        ]

        raise ValueError(
            "The model index contains duplicate model/"
            "abundance/time/line records:\n"
            f"{duplicate_rows.head(10)}"
        )

    missing_parameters = {
        parameter: int(
            model_index[
                parameter
            ].isna().sum()
        )
        for parameter in INDEX_PARAMETERS
        if model_index[
            parameter
        ].isna().any()
    }

    if missing_parameters:
        raise ValueError(
            "The model index contains missing parameter "
            f"values: {missing_parameters}."
        )


def print_model_index_summary(
    model_index: pd.DataFrame,
    output_file: str | Path,
    missing_model_info: list[str] | None = None,
) -> None:
    """Print a summary of a constructed model index."""

    print()
    print("SPAMMS model index")
    print("==================")
    print(f"Total rows: {len(model_index)}")
    print(
        "Unique models: "
        f"{model_index['model_id'].nunique()}"
    )
    print(
        "Unique lines: "
        f"{model_index['line_name'].nunique()}"
    )
    print(
        "Missing profile files: "
        f"{int((~model_index['profile_exists']).sum())}"
    )

    if missing_model_info:
        print(
            "Model directories missing model_info.txt: "
            f"{len(missing_model_info)}"
        )

    print()
    print("Parameter values:")

    for parameter in INDEX_PARAMETERS:
        values = sorted(
            model_index[
                parameter
            ].dropna().unique()
        )

        print(
            f"  {parameter}: {values}"
        )

    print()
    print(f"Saved model index to: {output_file}")


def _get_selected_lines(
    input_parameters: Mapping[str, Any],
) -> tuple[str, ...]:
    """Extract the line list used to calculate the grid."""

    selected_lines = input_parameters.get(
        "selected_line_list"
    )

    if selected_lines is None:
        selected_lines = input_parameters.get(
            "selected_he_line_list"
        )

    if selected_lines is None:
        raise ValueError(
            "Could not find selected_line_list "
            "in the grid input file."
        )

    lines = _ensure_list(
        selected_lines
    )

    prepared: list[str] = []

    for line in lines:
        if not isinstance(line, str):
            raise TypeError(
                "Every selected line must be a string."
            )

        line = line.strip()

        if not line:
            raise ValueError(
                "Selected line names cannot be empty."
            )

        prepared.append(line)

    if len(set(prepared)) != len(prepared):
        raise ValueError(
            "The selected line list contains duplicates."
        )

    return tuple(prepared)


def _parse_value(
    raw_value: str,
) -> Any:
    """Parse one input or model-information value."""

    if raw_value == "None":
        return None

    try:
        return ast.literal_eval(
            raw_value
        )
    except (ValueError, SyntaxError):
        return raw_value


def _ensure_list(
    value: Any,
) -> list[Any]:
    """Return a scalar or sequence as a list."""

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _model_number(
    model_directory: Path,
) -> int:
    """Extract the numerical identifier from Model_XXXX."""

    match = MODEL_DIRECTORY_PATTERN.fullmatch(
        model_directory.name
    )

    if match is None:
        raise ValueError(
            f"Invalid model-directory name: "
            f"{model_directory.name}"
        )

    return int(match.group(1))


def _stored_path(
    path: Path,
    index_directory: Path,
    relative_paths: bool,
) -> str:
    """Return a portable relative or absolute stored path."""

    path = path.resolve()

    if relative_paths:
        try:
            return str(
                path.relative_to(
                    index_directory
                )
            )
        except ValueError:
            # The path is outside the index directory.
            return str(path)

    return str(path)
