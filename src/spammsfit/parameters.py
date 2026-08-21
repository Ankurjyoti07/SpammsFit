"""Parameter-state management for SPAMMSFit."""

from __future__ import annotations
import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
MANAGED_PARAMETERS = ("ntriangles", "teff", "r_pole", "mass", "inclination", "v_crit_frac", "sigma_R", "sigma_T") # All numerical SPAMMS settings managed by ParameterSet.
FITTABLE_PARAMETERS = ("teff", "r_pole", "mass", "inclination", "v_crit_frac", "sigma_R", "sigma_T")
SPAMMS_LIST_PARAMETERS = { "inclination", "v_crit_frac", "sigma_R", "sigma_T"}
INTEGER_PARAMETERS = {"ntriangles"} # Parameters that must have integer values.

# Basic physical limits independent of any model-grid limits.
PHYSICAL_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "ntriangles": (1, None),
    "teff": (0.0, None),
    "r_pole": (0.0, None),
    "mass": (0.0, None),
    "inclination": (0.0, 90.0),
    "v_crit_frac": (0.0, 1.0),
    "sigma_R": (0.0, None),
    "sigma_T": (0.0, None),
}


@dataclass
class Parameter:
    """
    Store the fitting state of one SPAMMS parameter.
    name: Parameter name used in the SPAMMS input file.
    value: Current numerical value.
    fixed: Whether the parameter remains unchanged during fitting.
    bounds: Lower and upper limits for a free continuous parameter.
    values: Permitted values for a free discrete parameter.
    prior : Optional Bayesian prior description.
    """
    name: str
    value: float | int
    fixed: bool = True
    bounds: tuple[float, float] | None = None
    values: tuple[float | int, ...] | None = None
    prior: str | Mapping[str, Any] | None = None

    @property
    def is_free(self) -> bool:
        """Return whether this parameter is free."""
        return not self.fixed

    @property
    def is_continuous(self) -> bool:
        """Return whether this is a free continuous parameter."""
        return ( not self.fixed and self.bounds is not None and self.values is None)

    @property
    def is_discrete(self) -> bool:
        """Return whether this is a free discrete parameter."""
        return (not self.fixed and self.values is not None and self.bounds is None)

    def fix( self, value: float | int | None = None) -> None:
        """Fix the parameter, optionally at a new value."""
        if value is not None:
            self.value = value
        self.fixed = True
        self.bounds = None
        self.values = None
        self.prior = None

    def free_continuous(
        self,
        bounds: tuple[float, float],
        prior: str | Mapping[str, Any] | None = None,
    ) -> None:
        """Make the parameter free within continuous bounds."""

        self.fixed = False
        self.bounds = bounds
        self.values = None
        self.prior = prior

    def free_discrete(
        self,
        values: tuple[float | int, ...],
        prior: str | Mapping[str, Any] | None = None,
    ) -> None:
        """Make the parameter free over discrete permitted values."""

        self.fixed = False
        self.bounds = None
        self.values = values
        self.prior = prior


class ParameterSet:
    """
    Track fixed and free SPAMMS parameters.

    The class reads starting values from an existing SPAMMS input file.
    It does not modify that file or generate temporary input files.

    Parameters
    ----------
    input_file
        Path to the base SPAMMS input file.

    Examples
    --------
    Load the current SPAMMS settings:

    >>> parameters = ParameterSet(
    ...     input_file="input.txt",
    ... )

    Fix stellar parameters:

    >>> parameters.fix(
    ...     teff=40000,
    ...     r_pole=6.5,
    ...     mass=25,
    ... )

    Select continuous fitting parameters:

    >>> parameters.free(
    ...     inclination=(10, 90),
    ...     v_crit_frac=(0.1, 0.9),
    ...     sigma_R=(50, 300),
    ...     sigma_T=(50, 300),
    ... )
    """

    def __init__(
        self,
        input_file: str | Path,
    ) -> None:
        self.input_file = (
            Path(input_file)
            .expanduser()
            .resolve()
        )

        if not self.input_file.is_file():
            raise FileNotFoundError(
                f"SPAMMS input file not found: "
                f"{self.input_file}"
            )

        input_values = self._read_input_values(
            self.input_file,
        )

        self._parameters = self._create_parameters(
            input_values,
        )

        self._selected_lines = self._extract_selected_lines(
            input_values,
        )

        self.validate()

    @staticmethod
    def _read_input_values(
        input_file: Path,
    ) -> dict[str, Any]:
        """
        Read active assignments from a SPAMMS input file.

        Blank lines and lines beginning with '#' are ignored.
        """

        values: dict[str, Any] = {}

        text = input_file.read_text(
            encoding="utf-8",
        )

        for line_number, raw_line in enumerate(
            text.splitlines(),
            start=1,
        ):
            stripped = raw_line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            if "=" not in stripped:
                continue

            name, raw_value = stripped.split(
                "=",
                maxsplit=1,
            )

            name = name.strip()
            raw_value = raw_value.strip()

            if not name:
                raise ValueError(
                    f"Empty parameter name on line "
                    f"{line_number}."
                )

            if name in values:
                raise ValueError(
                    f"Duplicate active parameter {name!r} "
                    f"on line {line_number}."
                )

            values[name] = ParameterSet._parse_input_value(
                raw_value,
            )

        return values

    @staticmethod
    def _parse_input_value(
        raw_value: str,
    ) -> Any:
        """
        Parse a SPAMMS input value.

        Python-style numbers, lists and None are parsed directly.
        Bare strings and paths are retained as strings.
        """

        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return raw_value

    @staticmethod
    def _create_parameters(
        input_values: Mapping[str, Any],
    ) -> dict[str, Parameter]:
        """Create Parameter objects from parsed SPAMMS values."""

        parameters: dict[str, Parameter] = {}

        for name in MANAGED_PARAMETERS:
            if name not in input_values:
                raise ValueError(
                    f"Required parameter {name!r} is missing "
                    "from the SPAMMS input file."
                )

            value = ParameterSet._extract_parameter_value(
                name=name,
                value=input_values[name],
            )

            parameters[name] = Parameter(
                name=name,
                value=value,
                fixed=True,
            )

        return parameters

    @staticmethod
    def _extract_parameter_value(
        name: str,
        value: Any,
    ) -> float | int:
        """
        Extract and validate a numerical parameter value.

        SPAMMS stores inclination, v_crit_frac, sigma_R and sigma_T
        as one-element lists. Internally, SPAMMSFit stores them as
        numerical scalars.
        """

        if name in SPAMMS_LIST_PARAMETERS:
            if not isinstance(value, list):
                raise ValueError(
                    f"Parameter {name!r} must be stored as "
                    "a list in the SPAMMS input file."
                )

            if len(value) != 1:
                raise ValueError(
                    f"Parameter {name!r} must initially contain "
                    "exactly one value."
                )

            value = value[0]

        return ParameterSet._coerce_value(
            name=name,
            value=value,
        )

    @staticmethod
    def _extract_selected_lines(
        input_values: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Extract and validate selected_line_list."""

        if "selected_line_list" not in input_values:
            raise ValueError(
                "'selected_line_list' is missing from "
                "the SPAMMS input file."
            )

        lines = input_values["selected_line_list"]

        if not isinstance(lines, list):
            raise ValueError(
                "'selected_line_list' must be stored as "
                "a Python-style list."
            )

        return ParameterSet._prepare_line_names(
            lines,
        )

    @staticmethod
    def _coerce_value(
        name: str,
        value: Any,
    ) -> float | int:
        """Convert a parameter to its internal numerical type."""

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"Parameter {name!r} must be numerical; "
                f"received {type(value).__name__}."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"Parameter {name!r} must be finite."
            )

        if name in INTEGER_PARAMETERS:
            integer_value = int(value)

            if integer_value != value:
                raise ValueError(
                    f"Parameter {name!r} must have "
                    "an integer value."
                )

            return integer_value

        return float(value)

    def fix(
        self,
        **parameter_values: float | int,
    ) -> None:
        """
        Fix one or more parameters at supplied values.

        Examples
        --------
        >>> parameters.fix(
        ...     teff=40000,
        ...     mass=25,
        ...     r_pole=6.5,
        ... )
        """

        if not parameter_values:
            raise ValueError(
                "At least one parameter must be supplied "
                "to fix()."
            )

        for name, value in parameter_values.items():
            parameter = self._get_parameter(name)

            prepared_value = self._prepare_value(
                name=name,
                value=value,
            )

            parameter.fix(prepared_value)

        self.validate()

    def free(
        self,
        *,
        priors: Mapping[
            str,
            str | Mapping[str, Any],
        ] | None = None,
        **parameter_bounds: tuple[float, float],
    ) -> None:
        """
        Make parameters free within continuous bounds.

        This is the primary configuration for DEFit and BayesianFit.

        Parameters
        ----------
        priors
            Optional mapping of parameter names to Bayesian priors.
        **parameter_bounds
            Parameter names and their lower/upper limits.

        Examples
        --------
        >>> parameters.free(
        ...     inclination=(10, 90),
        ...     v_crit_frac=(0.1, 0.9),
        ...     sigma_R=(50, 300),
        ...     sigma_T=(50, 300),
        ... )

        Bayesian priors can optionally be supplied:

        >>> parameters.free(
        ...     inclination=(10, 90),
        ...     v_crit_frac=(0.1, 0.9),
        ...     priors={
        ...         "inclination": "isotropic",
        ...         "v_crit_frac": "uniform",
        ...     },
        ... )
        """

        if not parameter_bounds:
            raise ValueError(
                "At least one parameter must be supplied "
                "to free()."
            )

        priors = (
            {}
            if priors is None
            else dict(priors)
        )

        unknown_priors = (
            set(priors)
            - set(parameter_bounds)
        )

        if unknown_priors:
            raise ValueError(
                "Priors were supplied for parameters not "
                "included in this free() call: "
                f"{sorted(unknown_priors)}."
            )

        for name, bounds in parameter_bounds.items():
            self._ensure_fittable(name)

            lower, upper = self._prepare_bounds(
                name=name,
                bounds=bounds,
            )

            self._parameters[name].free_continuous(
                bounds=(lower, upper),
                prior=priors.get(name),
            )

        self.validate()

    def free_discrete(
        self,
        *,
        priors: Mapping[
            str,
            str | Mapping[str, Any],
        ] | None = None,
        **parameter_values: Iterable[float | int],
    ) -> None:
        """
        Make parameters free over specified discrete values.

        This representation is intended mainly for ChiSquareFit.

        Examples
        --------
        >>> parameters.free_discrete(
        ...     teff=[35000, 37500, 40000],
        ...     r_pole=[5.5, 6.0, 6.5],
        ...     mass=[20, 25, 30],
        ... )
        """

        if not parameter_values:
            raise ValueError(
                "At least one parameter must be supplied "
                "to free_discrete()."
            )

        priors = (
            {}
            if priors is None
            else dict(priors)
        )

        unknown_priors = (
            set(priors)
            - set(parameter_values)
        )

        if unknown_priors:
            raise ValueError(
                "Priors were supplied for parameters not "
                "included in this free_discrete() call: "
                f"{sorted(unknown_priors)}."
            )

        for name, values in parameter_values.items():
            self._ensure_fittable(name)

            prepared_values = tuple(
                self._prepare_value(
                    name=name,
                    value=value,
                )
                for value in values
            )

            if not prepared_values:
                raise ValueError(
                    f"Parameter {name!r} requires at least "
                    "one discrete value."
                )

            if len(set(prepared_values)) != len(
                prepared_values
            ):
                raise ValueError(
                    f"Parameter {name!r} contains duplicate "
                    "discrete values."
                )

            self._parameters[name].free_discrete(
                values=prepared_values,
                prior=priors.get(name),
            )

        self.validate()

    def set_lines(
        self,
        lines: Iterable[str],
    ) -> None:
        """
        Set the spectral lines to be generated and fitted.

        SpammsFit will normally synchronize this automatically with
        Spectrum.line_names.
        """

        self._selected_lines = (
            self._prepare_line_names(lines)
        )

    def get(
        self,
        name: str,
    ) -> Parameter:
        """Return one Parameter object."""

        return self._get_parameter(name)

    def current_values(
        self,
    ) -> dict[str, float | int]:
        """
        Return the current numerical value of every managed parameter.
        """

        return {
            name: parameter.value
            for name, parameter in self._parameters.items()
        }

    def with_values(
        self,
        **parameter_values: float | int,
    ) -> dict[str, float | int]:
        """
        Return current values with validated temporary overrides.

        This method does not modify parameter values, fixed/free states,
        bounds or priors. It is intended for explicit forward-model
        calculations such as ``SpammsFit.preview_model()``.
        """
        values = self.current_values()

        for name, value in parameter_values.items():
            values[name] = self._prepare_value(
                name=name,
                value=value,
            )

        return values

    def fixed_values(
        self,
    ) -> dict[str, float | int]:
        """Return currently fixed parameter values."""

        return {
            name: parameter.value
            for name, parameter in self._parameters.items()
            if parameter.fixed
        }

    def free_names(self) -> tuple[str, ...]:
        """
        Return free parameter names in a stable order.

        The order returned here defines the order of optimizer and
        sampler vectors.
        """

        return tuple(
            name
            for name in FITTABLE_PARAMETERS
            if self._parameters[name].is_free
        )

    def fixed_names(self) -> tuple[str, ...]:
        """Return fixed parameter names in a stable order."""

        return tuple(
            name
            for name in MANAGED_PARAMETERS
            if self._parameters[name].fixed
        )

    def free_parameters(self) -> tuple[Parameter, ...]:
        """Return free Parameter objects in vector order."""

        return tuple(
            self._parameters[name]
            for name in self.free_names()
        )

    def continuous_bounds(
        self,
    ) -> tuple[tuple[float, float], ...]:
        """
        Return bounds in free-parameter vector order.

        This is used directly by DEFit and BayesianFit.
        """

        bounds: list[tuple[float, float]] = []

        for parameter in self.free_parameters():
            if not parameter.is_continuous:
                raise ValueError(
                    f"Free parameter {parameter.name!r} is "
                    "discrete and has no continuous bounds."
                )

            if parameter.bounds is None:
                raise RuntimeError(
                    f"Continuous parameter {parameter.name!r} "
                    "has no bounds."
                )

            bounds.append(parameter.bounds)

        return tuple(bounds)

    def initial_vector(self) -> FloatArray:
        """
        Return current values in free-parameter vector order.

        These values may be used as an initial guess.
        """

        values = [
            float(parameter.value)
            for parameter in self.free_parameters()
        ]

        return np.asarray(
            values,
            dtype=np.float64,
        )

    def vector_to_values(
        self,
        theta: ArrayLike,
        *,
        check_bounds: bool = False,
    ) -> dict[str, float | int]:
        """
        Combine a free-parameter vector with all fixed values.

        Parameters
        ----------
        theta
            Values in the order returned by free_names().
        check_bounds
            Raise an error when a proposed value lies outside its
            configured bounds.

        Returns
        -------
        dict
            Complete numerical parameter mapping for one SPAMMS model.
        """

        theta_array = np.asarray(
            theta,
            dtype=np.float64,
        )

        if theta_array.ndim != 1:
            raise ValueError(
                "theta must be a one-dimensional array."
            )

        free_parameters = self.free_parameters()

        if theta_array.size != len(free_parameters):
            raise ValueError(
                f"theta contains {theta_array.size} values, "
                f"but {len(free_parameters)} free parameters "
                "are configured."
            )

        model_values = self.current_values()

        for parameter, value in zip(
            free_parameters,
            theta_array,
            strict=True,
        ):
            if not parameter.is_continuous:
                raise ValueError(
                    f"Parameter {parameter.name!r} is discrete "
                    "and cannot be assigned from a continuous vector."
                )

            if check_bounds and not self._value_in_bounds(
                parameter,
                value,
            ):
                raise ValueError(
                    f"Value {value} lies outside the bounds "
                    f"{parameter.bounds} for parameter "
                    f"{parameter.name!r}."
                )

            model_values[parameter.name] = (
                self._prepare_value(
                    name=parameter.name,
                    value=value,
                )
            )

        return model_values

    def vector_in_bounds(
        self,
        theta: ArrayLike,
    ) -> bool:
        """
        Return whether every continuous vector value lies within bounds.

        This method is deliberately lightweight because BayesianFit may
        call it many thousands of times.
        """

        theta_array = np.asarray(
            theta,
            dtype=np.float64,
        )

        free_parameters = self.free_parameters()

        if theta_array.ndim != 1:
            return False

        if theta_array.size != len(free_parameters):
            return False

        if not np.all(np.isfinite(theta_array)):
            return False

        for parameter, value in zip(
            free_parameters,
            theta_array,
            strict=True,
        ):
            if not parameter.is_continuous:
                return False

            if not self._value_in_bounds(
                parameter,
                value,
            ):
                return False

        return True

    def discrete_values(
        self,
    ) -> dict[str, tuple[float | int, ...]]:
        """Return permitted values for free discrete parameters."""

        discrete: dict[
            str,
            tuple[float | int, ...],
        ] = {}

        for parameter in self.free_parameters():
            if parameter.is_discrete:
                if parameter.values is None:
                    raise RuntimeError(
                        f"Discrete parameter {parameter.name!r} "
                        "has no permitted values."
                    )

                discrete[parameter.name] = (
                    parameter.values
                )

        return discrete

    def validate(self) -> None:
        """Validate the complete parameter configuration."""

        for name, parameter in self._parameters.items():
            self._validate_physical_value(
                name=name,
                value=parameter.value,
            )

            if parameter.fixed:
                if parameter.bounds is not None:
                    raise ValueError(
                        f"Fixed parameter {name!r} cannot "
                        "have bounds."
                    )

                if parameter.values is not None:
                    raise ValueError(
                        f"Fixed parameter {name!r} cannot "
                        "have discrete permitted values."
                    )

                continue

            if name not in FITTABLE_PARAMETERS:
                raise ValueError(
                    f"Parameter {name!r} cannot be fitted."
                )

            if (
                parameter.bounds is None
                and parameter.values is None
            ):
                raise ValueError(
                    f"Free parameter {name!r} requires either "
                    "continuous bounds or discrete values."
                )

            if (
                parameter.bounds is not None
                and parameter.values is not None
            ):
                raise ValueError(
                    f"Parameter {name!r} cannot be both "
                    "continuous and discrete."
                )

            if parameter.bounds is not None:
                lower, upper = parameter.bounds

                self._validate_physical_value(
                    name=name,
                    value=lower,
                )

                self._validate_physical_value(
                    name=name,
                    value=upper,
                )

                if lower >= upper:
                    raise ValueError(
                        f"Invalid bounds {parameter.bounds} "
                        f"for parameter {name!r}."
                    )

            if parameter.values is not None:
                for value in parameter.values:
                    self._validate_physical_value(
                        name=name,
                        value=value,
                    )

        if not self._selected_lines:
            raise ValueError(
                "At least one spectral line must be selected."
            )

    def summary(self) -> str:
        """Return a readable summary of the fitting configuration."""

        lines = [
            f"SPAMMS input file: {self.input_file}",
            (
                "Selected lines: "
                f"{', '.join(self._selected_lines)}"
            ),
            "Parameters:",
        ]

        for name in MANAGED_PARAMETERS:
            parameter = self._parameters[name]

            if parameter.fixed:
                state = (
                    f"fixed at {parameter.value}"
                )
            elif parameter.is_continuous:
                state = (
                    f"free continuous, "
                    f"bounds={parameter.bounds}"
                )
            else:
                state = (
                    f"free discrete, "
                    f"values={parameter.values}"
                )

            if parameter.prior is not None:
                state += (
                    f", prior={parameter.prior}"
                )

            lines.append(
                f"  {name}: {state}"
            )

        return "\n".join(lines)

    def _get_parameter(
        self,
        name: str,
    ) -> Parameter:
        """Return a parameter or raise an informative error."""

        try:
            return self._parameters[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown parameter {name!r}. Managed "
                f"parameters are: "
                f"{', '.join(MANAGED_PARAMETERS)}."
            ) from error

    @staticmethod
    def _ensure_fittable(
        name: str,
    ) -> None:
        """Ensure that a parameter may be made free."""

        if name not in MANAGED_PARAMETERS:
            raise KeyError(
                f"Unknown parameter {name!r}. Managed "
                f"parameters are: "
                f"{', '.join(MANAGED_PARAMETERS)}."
            )

        if name not in FITTABLE_PARAMETERS:
            raise ValueError(
                f"Parameter {name!r} is a model setting "
                "and cannot be fitted."
            )

    def _prepare_value(
        self,
        name: str,
        value: Any,
    ) -> float | int:
        """Convert and physically validate one supplied value."""

        if name not in MANAGED_PARAMETERS:
            raise KeyError(
                f"Unknown parameter {name!r}."
            )

        prepared_value = self._coerce_value(
            name=name,
            value=value,
        )

        self._validate_physical_value(
            name=name,
            value=prepared_value,
        )

        return prepared_value

    def _prepare_bounds(
        self,
        name: str,
        bounds: tuple[float, float],
    ) -> tuple[float, float]:
        """Validate continuous lower and upper bounds."""

        if not isinstance(bounds, (tuple, list)):
            raise TypeError(
                f"Bounds for {name!r} must be a tuple "
                "or list containing two values."
            )

        if len(bounds) != 2:
            raise ValueError(
                f"Bounds for {name!r} must contain "
                "exactly two values."
            )

        lower = self._prepare_value(
            name=name,
            value=bounds[0],
        )

        upper = self._prepare_value(
            name=name,
            value=bounds[1],
        )

        if lower >= upper:
            raise ValueError(
                f"The lower bound for {name!r} must be "
                "smaller than the upper bound."
            )

        return float(lower), float(upper)

    @staticmethod
    def _validate_physical_value(
        name: str,
        value: float | int,
    ) -> None:
        """Validate a value against basic physical limits."""

        lower, upper = PHYSICAL_LIMITS[name]

        if lower is not None and value < lower:
            raise ValueError(
                f"Parameter {name!r} must be >= {lower}; "
                f"received {value}."
            )

        if upper is not None and value > upper:
            raise ValueError(
                f"Parameter {name!r} must be <= {upper}; "
                f"received {value}."
            )

    @staticmethod
    def _value_in_bounds(
        parameter: Parameter,
        value: float,
    ) -> bool:
        """Check one continuous parameter value against its bounds."""

        if parameter.bounds is None:
            return False

        lower, upper = parameter.bounds

        return (
            np.isfinite(value)
            and lower < value < upper
        )

    @staticmethod
    def _prepare_line_names(
        lines: Iterable[str],
    ) -> tuple[str, ...]:
        """Validate and standardize selected line names."""

        if isinstance(lines, str):
            raise TypeError(
                "lines must be an iterable of line names, "
                "not a single string."
            )

        prepared: list[str] = []

        for line in lines:
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
                "At least one spectral line must be selected."
            )

        if len(set(prepared)) != len(prepared):
            raise ValueError(
                "The selected line list contains duplicate names."
            )

        return tuple(prepared)

    @property
    def selected_lines(self) -> tuple[str, ...]:
        """Return the selected SPAMMS line names."""

        return self._selected_lines

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return all managed parameter names."""

        return MANAGED_PARAMETERS

    @property
    def n_free(self) -> int:
        """Return the number of free fitting parameters."""

        return len(self.free_names())

    @property
    def n_fixed(self) -> int:
        """Return the number of fixed parameters."""

        return len(self.fixed_names())

    def __contains__(self, name: str) -> bool:
        """Return whether a parameter is managed."""

        return name in self._parameters

    def __getitem__(
        self,
        name: str,
    ) -> Parameter:
        """Provide dictionary-style parameter access."""

        return self._get_parameter(name)

    def __len__(self) -> int:
        """Return the number of managed parameters."""

        return len(self._parameters)

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"ParameterSet("
            f"input_file={str(self.input_file)!r}, "
            f"n_fixed={self.n_fixed}, "
            f"n_free={self.n_free}, "
            f"selected_lines={self._selected_lines!r}"
            f")"
        )
