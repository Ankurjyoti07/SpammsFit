from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Any
from spammsfit.core import SpammsFit

class BaseFit(ABC):
    """
    base class for SPAMMSFit fitting methods. BaseFit contains only functionality that generally applies to every fitting method.
    """

    method_name = "Base fit"
    def __init__(self, spamms_fit: SpammsFit) -> None:
        if not isinstance(spamms_fit, SpammsFit):
            raise TypeError("spamms_fit must be a SpammsFit instance.")
        self.spamms_fit = spamms_fit
        self._result: Any | None = None
        self._status = "not_started"
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._runtime: float | None = None
        self._failure: Exception | None = None

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the fitting method and return its result. Every fitting subclass must implement this method.
        """
        raise NotImplementedError

    def _start_run(self) -> None:
        """Record the beginning of a fitting calculation."""
        if self._status == "running":
            raise RuntimeError(f"{self.method_name} is already running.")
        self._result = None
        self._failure = None
        self._start_time = time.perf_counter()
        self._end_time = None
        self._runtime = None
        self._status = "running"

    def _finish_run(self, result: Any) -> Any:
        """Record successful completion and return the result."""
        if self._status != "running":
            raise RuntimeError("Cannot finish a fit that is not running.")
        self._end_time = time.perf_counter()
        self._runtime = self._end_time - self._start_time
        self._result = result
        self._status = "completed"
        return result

    def _fail_run(self, error: Exception) -> None:
        """Record an unsuccessful fitting calculation."""
        self._end_time = time.perf_counter()
        if self._start_time is not None:
            self._runtime = self._end_time - self._start_time
        self._failure = error
        self._status = "failed"

    def require_free_parameters(self) -> None:
        """Ensure that at least one parameter is free."""
        if self.spamms_fit.parameters.n_free == 0:
            raise ValueError(f"{self.method_name} requires at least one free parameter.")

    def require_continuous_parameters(self) -> None:
        """
        Ensure that every free parameter is continuous. This is required by DEFit and the initial BayesianFit.
        """
        self.require_free_parameters()
        discrete = [
            parameter.name
            for parameter in self.spamms_fit.parameters.free_parameters()
            if not parameter.is_continuous]
        if discrete:
            raise ValueError(f"{self.method_name} requires continuous free parameters. Discrete parameters found: {discrete}.")

    def require_discrete_parameters(self) -> None:
        """
        Ensure that every free parameter is discrete. This can be used by a grid-based ChiSquareFit configuration.
        """
        self.require_free_parameters()
        continuous = [
            parameter.name
            for parameter in self.spamms_fit.parameters.free_parameters()
            if not parameter.is_discrete]
        if continuous:
            raise ValueError(
                f"{self.method_name} requires discrete free parameters. Continuous parameters found: {continuous}.")

    def summary(self) -> str:
        """Return a summary of the fitting-method state."""
        runtime = (
            "not available"
            if self._runtime is None
            else f"{self._runtime:.3f} s")
        free_parameters = (", ".join(self.spamms_fit.parameters.free_names()) or "none")
        lines = [
            f"Method: {self.method_name}",
            f"Status: {self._status}",
            f"Runtime: {runtime}",
            f"Free parameters: {free_parameters}"]
        if self._failure is not None:
            lines.append(f"Failure: {self._failure}")
        return "\n".join(lines)

    @property
    def result(self) -> Any:
        """
        Return the completed result.
        Raises RuntimeError If the fit has not completed successfully.
        """
        if self._status != "completed":
            raise RuntimeError(f"{self.method_name} has no completed result. Current status: {self._status}.")
        return self._result

    @property
    def status(self) -> str:
        """Return the current fitting status."""
        return self._status

    @property
    def runtime(self) -> float | None:
        """Return the completed runtime in seconds."""
        return self._runtime

    @property
    def has_run(self) -> bool:
        """Return whether execution has been attempted."""
        return self._status in {"completed", "failed"}

    @property
    def succeeded(self) -> bool:
        """Return whether execution completed successfully."""
        return self._status == "completed"

    @property
    def failed(self) -> bool:
        """Return whether execution failed."""
        return self._status == "failed"

    @property
    def failure(self) -> Exception | None:
        """Return the captured failure, if any."""
        return self._failure

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{self.__class__.__name__}("
            f"status={self._status!r}, "
            f"n_free={self.spamms_fit.parameters.n_free})")
