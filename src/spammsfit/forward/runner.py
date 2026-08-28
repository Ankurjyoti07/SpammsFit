"""Execution of SPAMMS forward-model calculations."""

from __future__ import annotations
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from spammsfit.configuration import SpammsConfig
from spammsfit.forward.input import InputBuilder
from spammsfit.forward.output import ModelSpectra, read_model_spectra


class SpammsRunner:
    """
    Execute SPAMMS for concrete numerical parameter values.

    Parameters
    ----------
    config
        Runtime paths and subprocess settings.
    input_builder
        Builder initialized from the base SPAMMS input template.

    Notes
    -----
    Every call to run() creates a unique temporary directory. This makes
    simultaneous Bayesian or DE evaluations safe because each process
    receives an independent input file and output directory.
    """

    def __init__(self, config: SpammsConfig, input_builder: InputBuilder) -> None:
        self.config = config
        self.input_builder = input_builder
        self.config.temporary_directory.mkdir(parents=True,exist_ok=True)
        self._n_requests = 0
        self._n_successful = 0
        self._n_failed = 0
        self._total_runtime = 0.0
        self._total_spamms_runtime = 0.0
        self._last_runtime: float | None = None
        self._last_spamms_runtime: float | None = None

    def run(self,parameter_values: Mapping[str, float | int], selected_lines: Iterable[str]) -> ModelSpectra:
        """
        Calculate one multiline SPAMMS model.

        Parameters
        ----------
        parameter_values
            Complete numerical parameter mapping for one model.
        selected_lines
            All line names to calculate in this SPAMMS execution.

        Returns
        -------
        dict
            Mapping from line names to wavelength and flux arrays.
        """

        selected_lines = tuple(selected_lines)

        if not selected_lines:raise ValueError("At least one spectral line must be selected.")
        self._n_requests += 1
        run_start = time.perf_counter()
        spamms_runtime: float | None = None
        run_succeeded = False
        temporary_directory = Path(tempfile.mkdtemp(prefix="spamms_",dir=self.config.temporary_directory)).resolve()
        input_file = temporary_directory / "input.txt"
        output_directory = temporary_directory / "output"
        try:
            output_directory.mkdir()
            input_text = self.input_builder.build(parameter_values=parameter_values,selected_lines=selected_lines,output_directory=output_directory)
            input_file.write_text(input_text,encoding="utf-8")
            command = self.config.command(input_file=input_file)
            spamms_start = time.perf_counter()
            process = subprocess.run(command, cwd=self.config.spamms_directory,env=self.config.subprocess_environment(),
                stdout=(subprocess.DEVNULL if self.config.suppress_stdout else None), stderr=subprocess.PIPE, text=True,
                timeout=self.config.timeout, check=False)
            spamms_runtime = (time.perf_counter() - spamms_start)
            if process.returncode != 0:
                raise RuntimeError(self._format_failure_message( parameter_values=parameter_values, temporary_directory=temporary_directory,
                        return_code=process.returncode, stderr=process.stderr))
            models = read_model_spectra( output_directory=output_directory, selected_lines=selected_lines)
            run_succeeded = True
            self._n_successful += 1
            return models

        except subprocess.TimeoutExpired as error:
            raise RuntimeError(self._format_timeout_message( parameter_values=parameter_values, temporary_directory=temporary_directory, timeout=error.timeout, stderr=error.stderr)) from error
        finally:
            total_runtime = (time.perf_counter() - run_start)
            self._last_runtime = total_runtime
            self._total_runtime += total_runtime
            if spamms_runtime is not None:
                self._last_spamms_runtime = (spamms_runtime)
                self._total_spamms_runtime += ( spamms_runtime)
            if not run_succeeded:
                self._n_failed += 1
            should_remove = ( run_succeeded or not self.config.keep_failed_runs)
            if should_remove:
                shutil.rmtree(temporary_directory, ignore_errors=True)

    @staticmethod
    def _format_failure_message(
        parameter_values: Mapping[str, float | int],
        temporary_directory: Path,
        return_code: int,
        stderr: str | None,
    ) -> str:
        """Construct an informative SPAMMS failure message."""

        stderr = (
            stderr.strip()
            if stderr
            else "No stderr output was produced."
        )

        parameter_text = ", ".join(
            f"{name}={value}"
            for name, value
            in parameter_values.items()
        )

        return (
            f"SPAMMS failed with return code "
            f"{return_code}.\n"
            f"Parameters: {parameter_text}\n"
            f"Temporary directory: "
            f"{temporary_directory}\n"
            f"stderr:\n{stderr}"
        )

    @staticmethod
    def _format_timeout_message(
        parameter_values: Mapping[str, float | int],
        temporary_directory: Path,
        timeout: float | None,
        stderr: str | bytes | None,
    ) -> str:
        """Construct an informative timeout message."""

        if isinstance(stderr, bytes):
            stderr = stderr.decode(
                errors="replace",
            )

        stderr = (
            stderr.strip()
            if stderr
            else "No stderr output was produced."
        )

        parameter_text = ", ".join(
            f"{name}={value}"
            for name, value
            in parameter_values.items()
        )

        return (
            f"SPAMMS exceeded the timeout of "
            f"{timeout} seconds.\n"
            f"Parameters: {parameter_text}\n"
            f"Temporary directory: "
            f"{temporary_directory}\n"
            f"stderr:\n{stderr}"
        )

    def reset_statistics(self) -> None:
        """Reset accumulated execution statistics."""

        self._n_requests = 0
        self._n_successful = 0
        self._n_failed = 0

        self._total_runtime = 0.0
        self._total_spamms_runtime = 0.0
        self._last_runtime = None
        self._last_spamms_runtime = None

    def timing_summary(self) -> str:
        """Return a summary of accumulated execution timings."""

        mean_total = (
            self._total_runtime / self._n_requests
            if self._n_requests
            else 0.0
        )

        mean_spamms = (
            self._total_spamms_runtime
            / self._n_successful
            if self._n_successful
            else 0.0
        )

        mean_overhead = max(
            mean_total - mean_spamms,
            0.0,
        )

        return (
            f"Requested runs: {self._n_requests}\n"
            f"Successful runs: {self._n_successful}\n"
            f"Failed runs: {self._n_failed}\n"
            f"Mean total runtime: "
            f"{mean_total:.3f} s\n"
            f"Mean SPAMMS runtime: "
            f"{mean_spamms:.3f} s\n"
            f"Approximate mean overhead: "
            f"{mean_overhead:.3f} s"
        )

    @property
    def n_requests(self) -> int:
        """Return the number of requested model calculations."""

        return self._n_requests

    @property
    def n_successful(self) -> int:
        """Return the number of successful calculations."""

        return self._n_successful

    @property
    def n_failed(self) -> int:
        """Return the number of failed calculations."""

        return self._n_failed

    @property
    def last_runtime(self) -> float | None:
        """Return total runtime of the most recent request."""

        return self._last_runtime

    @property
    def last_spamms_runtime(self) -> float | None:
        """Return SPAMMS subprocess time for the most recent request."""

        return self._last_spamms_runtime

    @property
    def total_runtime(self) -> float:
        """Return accumulated total runtime."""

        return self._total_runtime

    @property
    def total_spamms_runtime(self) -> float:
        """Return accumulated SPAMMS subprocess runtime."""

        return self._total_spamms_runtime

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"SpammsRunner("
            f"n_requests={self._n_requests}, "
            f"n_successful={self._n_successful}, "
            f"n_failed={self._n_failed}"
            f")"
        )
