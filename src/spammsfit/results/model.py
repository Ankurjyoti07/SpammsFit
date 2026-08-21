"""Result container for an explicitly requested SPAMMS model preview."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spammsfit.core import EvaluationDetails
from spammsfit.results.base import BaseResult


class ModelResult(BaseResult):
    """
    Store one SPAMMS forward model and its comparison with the data.

    Parameters
    ----------
    evaluation
        Detailed model evaluation containing the supplied parameters,
        native and interpolated profiles, residuals and fit statistics.
    runtime
        Total time required to generate and evaluate the model.
    metadata
        Optional additional information about the preview calculation.

    Notes
    -----
    A ModelResult represents one user-requested forward model. It is not
    the result of an optimizer or sampler and contains no fitted parameter
    uncertainties.
    """

    def __init__(
        self,
        *,
        evaluation: EvaluationDetails,
        runtime: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            method="SPAMMS model preview",
            best_parameters=evaluation["parameters"],
            free_parameter_names=(),
            chi2=evaluation["chi2"],
            reduced_chi2=evaluation["reduced_chi2"],
            log_likelihood=evaluation["log_likelihood"],
            runtime=runtime,
            n_evaluations=1,
            evaluation=evaluation,
            metadata=metadata,
        )

    @property
    def parameters(self) -> dict[str, float | int]:
        """Return the parameter values used to generate the model."""
        return dict(self.best_parameters)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"ModelResult(chi2={self.chi2:.6f}, "
            f"reduced_chi2={self.reduced_chi2:.6f}, "
            f"lines={self.line_names!r})"
        )
