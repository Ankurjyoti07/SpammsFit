"""Runtime configuration for SPAMMSFit."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class SpammsConfig:
    """
    Store paths and execution settings used by SPAMMSFit.

    Parameters
    ----------
    spamms_directory
        Root directory containing spamms.py.
    temporary_directory
        Root directory in which temporary SPAMMS calculations are run.
    results_directory
        Directory used for permanent SPAMMSFit results.
    spamms_script
        Name or path of the SPAMMS executable Python script. When only
        a filename is supplied, it is interpreted relative to
        spamms_directory.
    python_executable
        Python executable used to launch SPAMMS. By default, the
        currently active Python interpreter is used.
    timeout
        Optional maximum runtime, in seconds, for one SPAMMS execution.
        None means no timeout.
    keep_failed_runs
        Preserve a temporary calculation directory when SPAMMS fails.
        This is useful for debugging but consumes storage.
    suppress_stdout
        Suppress standard SPAMMS terminal output during fitting.
    environment
        Optional additional environment variables supplied to SPAMMS.
    """

    spamms_directory: str | Path
    temporary_directory: str | Path
    results_directory: str | Path

    spamms_script: str | Path = "spamms.py"
    python_executable: str | Path = sys.executable

    timeout: float | None = None
    keep_failed_runs: bool = False
    suppress_stdout: bool = True

    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate configuration values."""

        self.spamms_directory = self._resolve_path(
            self.spamms_directory,
        )

        self.temporary_directory = self._resolve_path(
            self.temporary_directory,
        )

        self.results_directory = self._resolve_path(
            self.results_directory,
        )

        self.python_executable = self._resolve_executable(
            self.python_executable,
        )

        self.spamms_script = self._resolve_spamms_script(
            self.spamms_script,
        )

        self._validate_timeout()
        self._prepare_environment()
        self.validate()

    @staticmethod
    def _resolve_path(
        path: str | Path,
    ) -> Path:
        """Expand and convert a path to an absolute Path."""

        return (
            Path(path)
            .expanduser()
            .resolve()
        )

    def _resolve_spamms_script(
        self,
        script: str | Path,
    ) -> Path:
        """
        Resolve the SPAMMS script path.

        A filename such as 'spamms.py' is interpreted relative to the
        configured SPAMMS directory. A path containing parent
        directories is resolved directly.
        """

        script = Path(script).expanduser()

        if script.is_absolute():
            return script.resolve()

        candidate = (
            self.spamms_directory
            / script
        )

        return candidate.resolve()

    @staticmethod
    def _resolve_executable(
        executable: str | Path,
    ) -> Path:
        """Resolve the configured Python executable."""

        executable = Path(executable).expanduser()

        if executable.is_absolute():
            return executable.resolve()

        # sys.executable is normally absolute. A relative custom
        # executable is retained so subprocess can search PATH.
        return executable

    def _validate_timeout(self) -> None:
        """Validate the optional SPAMMS timeout."""

        if self.timeout is None:
            return

        self.timeout = float(self.timeout)

        if self.timeout <= 0.0:
            raise ValueError(
                "timeout must be positive or None."
            )

    def _prepare_environment(self) -> None:
        """Copy and validate additional environment variables."""

        if self.environment is None:
            self.environment = {}
            return

        environment = dict(self.environment)

        for name, value in environment.items():
            if not isinstance(name, str):
                raise TypeError(
                    "Environment-variable names must be strings."
                )

            if not isinstance(value, str):
                raise TypeError(
                    f"Environment value for {name!r} must "
                    "be a string."
                )

        self.environment = environment

    def validate(self) -> None:
        """Validate all configured paths and settings."""

        if not self.spamms_directory.is_dir():
            raise NotADirectoryError(
                "SPAMMS directory does not exist: "
                f"{self.spamms_directory}"
            )

        if not self.spamms_script.is_file():
            raise FileNotFoundError(
                "SPAMMS script does not exist: "
                f"{self.spamms_script}"
            )

        if self.python_executable.is_absolute():
            if not self.python_executable.is_file():
                raise FileNotFoundError(
                    "Python executable does not exist: "
                    f"{self.python_executable}"
                )

        if self.temporary_directory == self.results_directory:
            raise ValueError(
                "temporary_directory and results_directory "
                "must be different directories."
            )

        if self.temporary_directory == self.spamms_directory:
            raise ValueError(
                "temporary_directory cannot be the main "
                "SPAMMS directory."
            )

    def prepare_directories(self) -> None:
        """
        Create temporary and permanent results directories.

        Directory creation is explicit rather than occurring silently
        when the configuration object is initialized.
        """

        self.temporary_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def subprocess_environment(
        self,
    ) -> dict[str, str]:
        """
        Return the complete environment for a SPAMMS subprocess.

        The current process environment is copied, then any configured
        overrides are applied.
        """

        environment = os.environ.copy()
        environment.update(self.environment)

        return environment

    def set_thread_limits(
        self,
        *,
        omp: int | None = None,
        mkl: int | None = None,
        openblas: int | None = None,
    ) -> None:
        """
        Configure numerical-library thread limits for SPAMMS subprocesses.

        This can prevent CPU oversubscription when several Bayesian or
        DE workers run SPAMMS simultaneously.

        Parameters
        ----------
        omp
            Value assigned to OMP_NUM_THREADS.
        mkl
            Value assigned to MKL_NUM_THREADS.
        openblas
            Value assigned to OPENBLAS_NUM_THREADS.
        """

        thread_settings = {
            "OMP_NUM_THREADS": omp,
            "MKL_NUM_THREADS": mkl,
            "OPENBLAS_NUM_THREADS": openblas,
        }

        updated_environment = dict(self.environment)

        for name, value in thread_settings.items():
            if value is None:
                continue

            value = int(value)

            if value < 1:
                raise ValueError(
                    f"{name} must be at least 1."
                )

            updated_environment[name] = str(value)

        self.environment = updated_environment

    def command(
        self,
        input_file: str | Path,
    ) -> list[str]:
        """
        Construct the command used to run SPAMMS.

        Parameters
        ----------
        input_file
            Concrete temporary SPAMMS input file.

        Returns
        -------
        list of str
            Command suitable for subprocess.run().
        """

        input_file = (
            Path(input_file)
            .expanduser()
            .resolve()
        )

        return [
            str(self.python_executable),
            str(self.spamms_script),
            "-i",
            str(input_file),
        ]

    def summary(self) -> str:
        """Return a human-readable configuration summary."""

        timeout = (
            "None"
            if self.timeout is None
            else f"{self.timeout:g} seconds"
        )

        environment = (
            "none"
            if not self.environment
            else ", ".join(
                f"{name}={value}"
                for name, value in sorted(
                    self.environment.items()
                )
            )
        )

        return (
            f"SPAMMS directory: {self.spamms_directory}\n"
            f"SPAMMS script: {self.spamms_script}\n"
            f"Python executable: {self.python_executable}\n"
            f"Temporary directory: "
            f"{self.temporary_directory}\n"
            f"Results directory: {self.results_directory}\n"
            f"Timeout: {timeout}\n"
            f"Keep failed runs: {self.keep_failed_runs}\n"
            f"Suppress stdout: {self.suppress_stdout}\n"
            f"Environment overrides: {environment}"
        )

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"SpammsConfig("
            f"spamms_directory="
            f"{str(self.spamms_directory)!r}, "
            f"temporary_directory="
            f"{str(self.temporary_directory)!r}, "
            f"results_directory="
            f"{str(self.results_directory)!r}"
            f")"
        )
