"""Construction of temporary SPAMMS input files."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


# Parameters that SPAMMS expects as one-element lists.
SPAMMS_LIST_PARAMETERS = {
    "inclination",
    "v_crit_frac",
    "sigma_R",
    "sigma_T",
}

# Numerical model parameters currently controlled by SPAMMSFit.
MODEL_PARAMETERS = {
    "ntriangles",
    "teff",
    "r_pole",
    "mass",
    "inclination",
    "v_crit_frac",
    "sigma_R",
    "sigma_T",
}

# Input fields that are always controlled during a model evaluation.
REQUIRED_RUNTIME_FIELDS = {
    "path_to_obs_spectra",
    "output_directory",
    "selected_line_list",
}


class InputBuilder:
    """
    Build concrete SPAMMS input text from a base template.

    Parameters
    ----------
    template_file
        Path to the base SPAMMS input file.

    Notes
    -----
    The template is read once during initialization. Each call to
    build() returns a short-lived string containing the concrete
    parameter values for one SPAMMS evaluation.

    This class does not modify the original input file and does not
    retain per-evaluation input text.
    """

    def __init__(
        self,
        template_file: str | Path,
    ) -> None:
        self.template_file = (
            Path(template_file)
            .expanduser()
            .resolve()
        )

        if not self.template_file.is_file():
            raise FileNotFoundError(
                f"SPAMMS input template not found: "
                f"{self.template_file}"
            )

        self._template = self.template_file.read_text(
            encoding="utf-8",
        )

        self._active_assignments = (
            self._find_active_assignments(
                self._template,
            )
        )

        self._validate_template()

    def build(
        self,
        parameter_values: Mapping[str, float | int],
        selected_lines: Iterable[str],
        output_directory: str | Path,
    ) -> str:
        """
        Build a complete numerical SPAMMS input for one evaluation.

        Parameters
        ----------
        parameter_values
            Complete numerical mapping containing all SPAMMSFit model
            parameters.
        selected_lines
            Names of all lines SPAMMS should calculate.
        output_directory
            Temporary output directory for this evaluation.

        Returns
        -------
        str
            Complete SPAMMS input text.

        Notes
        -----
        path_to_obs_spectra is always set to None. This prevents SPAMMS
        from performing its own chi-square calculation.
        """

        prepared_parameters = self._prepare_parameters(
            parameter_values,
        )

        prepared_lines = self._prepare_lines(
            selected_lines,
        )

        prepared_output_directory = (
            self._prepare_output_directory(
                output_directory,
            )
        )

        replacements: dict[str, str] = {
            "path_to_obs_spectra": "None",
            "output_directory": (
                f"{prepared_output_directory}/"
            ),
            "selected_line_list": repr(
                list(prepared_lines)
            ),
        }

        for name, value in prepared_parameters.items():
            replacements[name] = self._format_parameter(
                name=name,
                value=value,
            )

        input_text = self._template

        for name, value in replacements.items():
            input_text = self._replace_parameter(
                text=input_text,
                name=name,
                value=value,
            )

        return input_text

    def _validate_template(self) -> None:
        """Ensure that all required input fields exist exactly once."""

        required = (
            MODEL_PARAMETERS
            | REQUIRED_RUNTIME_FIELDS
        )

        missing = required - set(
            self._active_assignments
        )

        if missing:
            raise ValueError(
                "The SPAMMS input template is missing required "
                f"active assignments: {sorted(missing)}."
            )

        duplicates = {
            name: count
            for name, count
            in self._active_assignments.items()
            if count > 1
        }

        if duplicates:
            raise ValueError(
                "The SPAMMS input template contains duplicate "
                f"active assignments: {duplicates}."
            )

    @staticmethod
    def _find_active_assignments(
        text: str,
    ) -> dict[str, int]:
        """
        Count active assignments in the input template.

        Commented assignments such as:

        #selected_line_list = [...]

        are ignored.
        """

        assignments: dict[str, int] = {}

        for raw_line in text.splitlines():
            stripped = raw_line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            if "=" not in stripped:
                continue

            name = stripped.split(
                "=",
                maxsplit=1,
            )[0].strip()

            assignments[name] = (
                assignments.get(name, 0) + 1
            )

        return assignments

    @staticmethod
    def _prepare_parameters(
        parameter_values: Mapping[str, float | int],
    ) -> dict[str, float | int]:
        """Validate the supplied model-parameter mapping."""

        supplied = set(parameter_values)
        missing = MODEL_PARAMETERS - supplied
        unknown = supplied - MODEL_PARAMETERS

        if missing:
            raise ValueError(
                "Missing numerical model parameters: "
                f"{sorted(missing)}."
            )

        if unknown:
            raise ValueError(
                "Unknown numerical model parameters: "
                f"{sorted(unknown)}."
            )

        prepared: dict[str, float | int] = {}

        for name in MODEL_PARAMETERS:
            value = parameter_values[name]

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"Parameter {name!r} must be numerical; "
                    f"received {type(value).__name__}."
                )

            prepared[name] = value

        return prepared

    @staticmethod
    def _prepare_lines(
        selected_lines: Iterable[str],
    ) -> tuple[str, ...]:
        """Validate selected SPAMMS line names."""

        if isinstance(selected_lines, str):
            raise TypeError(
                "selected_lines must be an iterable of names, "
                "not one string."
            )

        prepared: list[str] = []

        for line in selected_lines:
            if not isinstance(line, str):
                raise TypeError(
                    "Every selected line name must be a string."
                )

            line = line.strip()

            if not line:
                raise ValueError(
                    "Selected line names cannot be empty."
                )

            prepared.append(line)

        if not prepared:
            raise ValueError(
                "At least one line must be selected."
            )

        if len(set(prepared)) != len(prepared):
            raise ValueError(
                "selected_lines contains duplicate names."
            )

        return tuple(prepared)

    @staticmethod
    def _prepare_output_directory(
        output_directory: str | Path,
    ) -> str:
        """Resolve and format a temporary output path."""

        output_directory = (
            Path(output_directory)
            .expanduser()
            .resolve()
        )

        return str(output_directory).rstrip("/")

    @staticmethod
    def _format_parameter(
        name: str,
        value: float | int,
    ) -> str:
        """Format a numerical value using SPAMMS input syntax."""

        if name in SPAMMS_LIST_PARAMETERS:
            return f"[{value}]"

        return str(value)

    @staticmethod
    def _replace_parameter(
        text: str,
        name: str,
        value: str,
    ) -> str:
        """
        Replace one active assignment in the SPAMMS template.

        Leading spacing and alignment before the original value are
        preserved. Commented assignments are not matched.
        """

        pattern = re.compile(
            rf"^(?P<prefix>\s*"
            rf"{re.escape(name)}"
            rf"\s*=\s*).*$",
            flags=re.MULTILINE,
        )

        new_text, count = pattern.subn(
            lambda match: (
                f"{match.group('prefix')}{value}"
            ),
            text,
            count=1,
        )

        if count != 1:
            raise ValueError(
                f"Could not uniquely replace active parameter "
                f"{name!r} in the SPAMMS input template."
            )

        return new_text

    @property
    def template_text(self) -> str:
        """
        Return the original template text.

        This is primarily useful for inspection and testing.
        """

        return self._template

    @property
    def active_assignments(
        self,
    ) -> tuple[str, ...]:
        """Return names of active template assignments."""

        return tuple(
            self._active_assignments
        )

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"InputBuilder("
            f"template_file="
            f"{str(self.template_file)!r}"
            f")"
        )
