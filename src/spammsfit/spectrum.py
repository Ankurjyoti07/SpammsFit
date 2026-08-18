"""Observed-spectrum handling for SPAMMSFit."""

from __future__ import annotations
from collections.abc import Iterator
from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

class LineData(NamedTuple):
    """
    Array views associated with one selected fitting line.
    This is a lightweight tuple, not a separate spectral-line class.
    """
    wavelength: FloatArray
    flux: FloatArray
    uncertainty: FloatArray

class FittingData(NamedTuple):
    """
    Concatenated arrays for all selected fitting regions.
    """
    wavelength: FloatArray
    flux: FloatArray
    uncertainty: FloatArray
    line_name: NDArray[np.str_]


class Spectrum:
    """
    store an observed spectrum and select wavelength regions for fitting.
    wavelength:  One-dimensional wavelength array. Values must be finite and strictly increasing.
    flux : One-dimensional observed flux array.
    uncertainty : Either one positive uncertainty value for the complete spectrum or an array containing one uncertainty per wavelength pixel.
    name : Optional identifier for the observation.
    example:
    >>> spectrum = Spectrum(wavelength=wave, flux=flux, uncertainty=1 / 500)
    add a line:
    >>> spectrum.add_line(name="HEII4200", fitting_range=(4175, 4225))
    >>> spectrum.add_line(name="HEI4026",fitting_range=(4000, 4050))
    remove a selected line:
    >>> spectrum.remove_line("HEI4026")
    """

    def __init__(self, wavelength: ArrayLike, flux: ArrayLike, uncertainty: ArrayLike | float, name: str | None = None) -> None:
        self.name = name
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.flux = np.asarray( flux, dtype=np.float64)
        self.uncertainty = self._prepare_uncertainty( uncertainty)
        self._validate_spectrum()

        # Maps each line name to:
        # ((lower wavelength, upper wavelength), array slice)
        self._lines: dict[str, tuple[tuple[float, float], slice]] = {}
        # Cached concatenated fitting data. It is created only when first
        # requested and invalidated whenever a line is added or removed.
        self._fitting_data_cache: FittingData | None = None

    def _prepare_uncertainty(self, uncertainty: ArrayLike | float) -> FloatArray:
        """
        Convert scalar or array uncertainties to a full float array.
        """
        uncertainty_array = np.asarray(uncertainty, dtype=np.float64)
        if uncertainty_array.ndim == 0:
            return np.full( self.wavelength.shape, float(uncertainty_array), dtype=np.float64)
        if uncertainty_array.ndim != 1:
            raise ValueError("uncertainty must be a scalar or a one-dimensional array.")
        return uncertainty_array

    def _validate_spectrum(self) -> None:
        """Validate the observed wavelength, flux and uncertainty arrays."""
        if self.wavelength.ndim != 1:
            raise ValueError("wavelength must be a one-dimensional array.")
        if self.flux.ndim != 1:
            raise ValueError("flux must be a one-dimensional array.")
        if self.uncertainty.ndim != 1:
            raise ValueError("uncertainty must be a one-dimensional array.")
        if self.wavelength.size == 0:
            raise ValueError("The observed spectrum cannot be empty.")
        if self.flux.shape != self.wavelength.shape:
            raise ValueError("wavelength and flux must have identical shapes.")
        if self.uncertainty.shape != self.flux.shape:
            raise ValueError("uncertainty must be a scalar or have the same shape as flux.")
        if not np.all(np.isfinite(self.wavelength)):
            raise ValueError("wavelength contains non-finite values.")
        if not np.all(np.isfinite(self.flux)):
            raise ValueError("flux contains non-finite values.")
        if not np.all(np.isfinite(self.uncertainty)):
            raise ValueError("uncertainty contains non-finite values.")
        if not np.all(np.diff(self.wavelength) > 0.0):
            raise ValueError("wavelength must be strictly increasing.")
        if np.any(self.uncertainty <= 0.0):
            raise ValueError("All uncertainty values must be positive.")

    def add_line(self, name: str, fitting_range: tuple[float, float], *, overwrite: bool = False) -> None:
        """
        add a line and wavelength bounds for fitting
        name: spectral line species -> same as FASTWIND line names.
        fitting_range: lower and upper wavelength limits.
        overwrite: replace an existing line with the same name when True.
        corresponding array slice is calculated once when the line is added and reused by all fitting evaluations.
        """
        name = self._validate_line_name(name)
        lower, upper = self._validate_fitting_range(fitting_range)
        if name in self._lines and not overwrite:
            raise ValueError(f"Line {name!r} already exists. use overwrite=True to replace it.")
        start = int(np.searchsorted(self.wavelength,lower,side="left"))
        stop = int(np.searchsorted(self.wavelength,upper,side="right"))
        if start == stop:
            raise ValueError(f"No observed wavelength pixels lie within the fitting range ({lower}, {upper}).")
        
        new_slice = slice(start, stop)
        self._check_for_overlap(name=name, new_slice=new_slice, overwrite=overwrite)
        self._lines[name] = ((lower, upper), new_slice)
        self._invalidate_cache()

    def remove_line(self, name: str) -> None:
        """
        remove a fitting region by its line name. original wavelength, flux and uncertainty arrays are not modified.
        """

        try:
            del self._lines[name]
        except KeyError as error:
            available = ", ".join(self._lines) or "none"
            raise KeyError(f"Unknown line {name!r}. Only available lines: {available}.") from error
        self._invalidate_cache()

    def clear_lines(self) -> None:
        """remove all selected fitting regions."""
        self._lines.clear()
        self._invalidate_cache()

    def get_line(self, name: str) -> LineData:
        """
        return array views for one selected line.
        returns: LineData : lightweight tuple containing wavelength, flux and uncertainty arrays.
        >>> wave, flux, error = spectrum.get_line("HEII4200")
        """
        try:
            _, line_slice = self._lines[name]
        except KeyError as error:
            available = ", ".join(self._lines) or "none"
            raise KeyError(f"Unknown line {name!r}. Available lines: {available}.") from error
        return LineData( wavelength=self.wavelength[line_slice], flux=self.flux[line_slice], uncertainty=self.uncertainty[line_slice])

    def iter_lines(self) -> Iterator[tuple[str, LineData]]:
        """
        Iterate over all selected fitting regions.
        >>> for name, data in spectrum.iter_lines():
                print(name, data.wavelength.size)
        """
        for name in self._lines:
            yield name, self.get_line(name)

    def get_fitting_data(self) -> FittingData:
        """
        return concatenated arrays for all selected fitting regions. The concatenated arrays are constructed only on the first call.
        Later calls return the cached arrays unless the selected lines have changed.
        """
        if not self._lines:
            raise RuntimeError("No spectral lines have been selected. Use add_line() before requesting fitting data.")
        if self._fitting_data_cache is not None:
            return self._fitting_data_cache

        wavelengths: list[FloatArray] = []
        fluxes: list[FloatArray] = []
        uncertainties: list[FloatArray] = []
        line_names: list[NDArray[np.str_]] = []
        for name, data in self.iter_lines():
            wavelengths.append(data.wavelength)
            fluxes.append(data.flux)
            uncertainties.append(data.uncertainty)
            line_names.append(np.full( data.wavelength.size, name, dtype=np.str_))
        fitting_data = FittingData(
            wavelength=np.concatenate(wavelengths),
            flux=np.concatenate(fluxes),
            uncertainty=np.concatenate(uncertainties),
            line_name=np.concatenate(line_names))
        self._make_arrays_read_only(fitting_data)
        self._fitting_data_cache = fitting_data
        return fitting_data

    def get_fitting_range(self, name: str) -> tuple[float, float]:
        """Return the requested wavelength limits for one line
        """
        try:
            fitting_range, _ = self._lines[name]
        except KeyError as error:
            available = ", ".join(self._lines) or "none"
            raise KeyError( f"Unknown line {name!r}. Available lines: {available}.") from error
        return fitting_range

    def summary(self) -> str:
        """Return a human-readable summary of the spectrum.
        """
        lines = [(
                f"  {name}: "
                f"{fitting_range[0]:.4f}–"
                f"{fitting_range[1]:.4f} "
                f"({line_slice.stop - line_slice.start} pixels)")
            for name, (fitting_range, line_slice)
            in self._lines.items()]
        line_summary = ("\n".join(lines) if lines else "  No lines selected")
        spectrum_name = ( self.name if self.name is not None else "Unnamed spectrum")
        return (
            f"Spectrum: {spectrum_name}\n"
            f"Total pixels: {self.size}\n"
            f"Selected lines: {self.n_lines}\n"
            f"Fitting pixels: {self.n_fitting_pixels}\n"
            f"{line_summary}")

    def _check_for_overlap(self, name: str, new_slice: slice, overwrite: bool) -> None:
        """
        Prevent overlapping line selections from double-counting pixels.
        """
        for existing_name, (_, existing_slice) in self._lines.items():
            if overwrite and existing_name == name:
                continue
            overlap_exists = ( new_slice.start < existing_slice.stop and existing_slice.start < new_slice.stop)
            if overlap_exists:
                raise ValueError(f"Line {name!r} overlaps with existing line "
                    f"{existing_name!r}. Overlapping fitting regions would count the same pixels more than once.")

    @staticmethod
    def _validate_line_name(name: str) -> str:
        """Validate and standardize a line name."""
        if not isinstance(name, str):
            raise TypeError("The line name must be a string.")
        name = name.strip()
        if not name:
            raise ValueError("The line name cannot be empty.")
        return name

    @staticmethod
    def _validate_fitting_range(fitting_range: tuple[float, float]) -> tuple[float, float]:
        """Validate a requested wavelength range."""
        if len(fitting_range) != 2:
            raise ValueError("fitting_range must contain exactly two values.")
        lower = float(fitting_range[0])
        upper = float(fitting_range[1])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("The fitting-range limits must be finite.")
        if lower >= upper:
            raise ValueError("The lower fitting limit must be smaller than the upper fitting limit.")
        return lower, upper

    def _invalidate_cache(self) -> None:
        """Discard cached fitting arrays after a selection changes."""

        self._fitting_data_cache = None

    @staticmethod
    def _make_arrays_read_only(
        fitting_data: FittingData,
    ) -> None:
        """Prevent accidental modification of cached fitting arrays."""

        fitting_data.wavelength.flags.writeable = False
        fitting_data.flux.flags.writeable = False
        fitting_data.uncertainty.flags.writeable = False
        fitting_data.line_name.flags.writeable = False

    @property
    def line_names(self) -> tuple[str, ...]:
        """Return selected line names in insertion order."""

        return tuple(self._lines)

    @property
    def line_ranges(
        self,
    ) -> dict[str, tuple[float, float]]:
        """Return a copy of the selected line-name/range mapping."""

        return {
            name: fitting_range
            for name, (fitting_range, _) in self._lines.items()
        }

    @property
    def size(self) -> int:
        """Return the number of pixels in the complete spectrum."""

        return int(self.wavelength.size)

    @property
    def n_lines(self) -> int:
        """Return the number of selected fitting regions."""

        return len(self._lines)

    @property
    def n_fitting_pixels(self) -> int:
        """Return the total number of selected fitting pixels."""

        return sum(
            line_slice.stop - line_slice.start
            for _, line_slice in self._lines.values()
        )

    def __contains__(self, name: str) -> bool:
        """Return whether a named line has been selected."""

        return name in self._lines

    def __len__(self) -> int:
        """Return the number of pixels in the complete spectrum."""

        return self.size

    def __repr__(self) -> str:
        """Return a concise representation of the spectrum."""

        return (
            f"Spectrum("
            f"name={self.name!r}, "
            f"size={self.size}, "
            f"n_lines={self.n_lines}, "
            f"n_fitting_pixels={self.n_fitting_pixels}"
            f")"
        )
