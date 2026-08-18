"""Bayesian inference with emcee for SPAMMSFit."""
from __future__ import annotations
import multiprocessing as mp
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import emcee
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm.auto import tqdm
from spammsfit.core import SpammsFit
from spammsfit.fits.base import BaseFit
from spammsfit.parameters import Parameter
from spammsfit.results.bayesian import BayesianResult

# Each multiprocessing worker receives its own SpammsFit instance once
# when the worker starts. This avoids repeatedly serializing the full
# fitting setup for every likelihood request.
_WORKER_SPAMMS_FIT: SpammsFit | None = None

def _initialize_bayesian_worker(spamms_fit: SpammsFit) -> None:
    """Initialize one multiprocessing likelihood worker."""
    global _WORKER_SPAMMS_FIT
    _WORKER_SPAMMS_FIT = spamms_fit

def _worker_log_probability(theta: ArrayLike) -> float:
    """Evaluate log probability inside a multiprocessing worker."""
    if _WORKER_SPAMMS_FIT is None:
        raise RuntimeError("Bayesian worker was not initialized.")
    return _calculate_log_probability(theta=theta, spamms_fit=_WORKER_SPAMMS_FIT)

def _calculate_log_probability(theta: ArrayLike, spamms_fit: SpammsFit) -> float:
    """Calculate log prior plus multiline log likelihood."""
    theta_array = np.asarray(theta, dtype=np.float64)
    log_prior = _calculate_log_prior(theta=theta_array, spamms_fit=spamms_fit)
    if not np.isfinite(log_prior):
        return -np.inf
    log_likelihood = spamms_fit.log_likelihood(theta_array)
    return float(log_prior + log_likelihood)

def _calculate_log_prior(theta: np.ndarray, spamms_fit: SpammsFit) -> float:
    """
    Calculate the configured prior for one parameter vector. All free parameters are truncated by their configured bounds.
    """
    parameters = spamms_fit.parameters
    if not parameters.vector_in_bounds(theta):
        return -np.inf
    total_log_prior = 0.0
    for parameter, value in zip(parameters.free_parameters(), theta, strict=True):
        total_log_prior += _parameter_log_prior( parameter=parameter, value=float(value))
    return float(total_log_prior)

def _parameter_log_prior(parameter: Parameter, value: float) -> float:
    """
    Calculate the prior contribution from one parameter.
    Supported priors for now:
    - None or uniform
    - isotropic for inclination
    - a Gaussian mapping with type, mean and sigma
    """
    prior = parameter.prior
    if prior is None or prior == "uniform":
        return 0.0
    if prior == "isotropic":
        if parameter.name != "inclination":
            raise ValueError("The isotropic prior is only valid for inclination.")
        sine_inclination = np.sin(np.deg2rad(value))
        if sine_inclination <= 0.0:
            return -np.inf
        return float(np.log(sine_inclination))

    if isinstance(prior, Mapping):
        prior_type = str(prior.get("type", "")).strip().lower()
        if prior_type == "uniform":
            return 0.0
        if prior_type == "isotropic":
            if parameter.name != "inclination":
                raise ValueError("The isotropic prior is only valid for inclination.")
            sine_inclination = np.sin(np.deg2rad(value))
            if sine_inclination <= 0.0:
                return -np.inf
            return float(np.log(sine_inclination))

        if prior_type == "gaussian":
            if "mean" not in prior:
                raise ValueError(f"Gaussian prior for {parameter.name!r} requires 'mean'.")
            if "sigma" not in prior:
                raise ValueError(f"Gaussian prior for {parameter.name!r} requires 'sigma'.")
            mean = float(prior["mean"])
            sigma = float(prior["sigma"])
            if sigma <= 0.0:
                raise ValueError(f"Gaussian prior sigma for {parameter.name!r} must be positive.")
            normalized_difference = (value - mean) / sigma
            return float(-0.5 * normalized_difference**2 - np.log(sigma * np.sqrt(2.0 * np.pi)))
        raise ValueError(
            f"Unknown prior type {prior_type!r} for parameter {parameter.name!r}.")
    raise ValueError(f"Unsupported prior {prior!r} for parameter {parameter.name!r}.")


class BayesianFit(BaseFit):
    """
    Fit continuous SPAMMS parameters using emcee. spamms_fit: Configured common SPAMMSFit calculation.
    The sampler dimension, parameter names, bounds and priors are obtained from ParameterSet. 
    The same class therefore supports any permitted subset of free continuous parameters.
    """
    method_name = "Bayesian inference"
    def __init__(self, spamms_fit: SpammsFit) -> None:
        super().__init__(spamms_fit=spamms_fit)

    def run(
        self,
        *,
        nsteps: int = 1000,
        nwalkers: int | None = None,
        ncores: int = 1,
        initial_guess: ArrayLike | None = None,
        burnin: int | None = None,
        thin: int = 1,
        seed: int | None = None,
        backend_file: str | Path | None = None,
        overwrite_backend: bool = False,
        progress: bool = True,
        show_recommendations: bool = True) -> BayesianResult:
        """
        Run Bayesian inference.
        Parameters
        ----------
        nsteps
            Number of MCMC steps per walker.
        nwalkers
            Number of ensemble walkers. When omitted, four times the
            number of free parameters is used.
        ncores
            Number of multiprocessing workers.
        initial_guess
            Optional initial free-parameter vector. One walker is placed
            exactly at this position; the remaining walkers are drawn
            uniformly within the parameter bounds.
        burnin
            Number of initial steps discarded from posterior summaries.
            By default, half the chain is discarded.
        thin
            Retain every ``thin``-th sample after burn-in.
        seed
            Random-number seed used to initialize the walkers.
        backend_file
            HDF5 chain path. By default,
            ``config.results_directory / "chain.h5"`` is used.
        overwrite_backend
            Reset and overwrite an existing HDF5 backend.
        progress
            Display a tqdm progress bar over MCMC steps.
        show_recommendations
            Print walker/core guidance before sampling.

        Returns
        -------
        BayesianResult
            Posterior samples, diagnostics and representative model.
        """

        self._start_run()
        run_start = time.perf_counter()
        try:
            self.require_continuous_parameters()
            settings = self._prepare_settings(
                nsteps=nsteps,
                nwalkers=nwalkers,
                ncores=ncores,
                initial_guess=initial_guess,
                burnin=burnin,
                thin=thin,
                seed=seed,
                backend_file=backend_file,
                overwrite_backend=overwrite_backend,
                progress=progress,
                show_recommendations=show_recommendations)

            if settings["show_recommendations"]:
                self._print_resource_guidance(settings)
            self._prepare_subprocess_threads(ncores=settings["ncores"])
            rng = np.random.default_rng(settings["seed"])
            initial_positions = self._initialize_walkers(
                rng=rng,
                nwalkers=settings["nwalkers"],
                initial_guess=settings["initial_guess"])
            backend = self._prepare_backend(
                backend_file=settings["backend_file"],
                nwalkers=settings["nwalkers"],
                ndim=settings["ndim"],
                overwrite=settings["overwrite_backend"])
            if settings["ncores"] == 1:
                sampler = emcee.EnsembleSampler(
                    settings["nwalkers"],
                    settings["ndim"],
                    self._serial_log_probability,
                    backend=backend)
                self._sample_with_progress(
                    sampler=sampler,
                    initial_positions=initial_positions,
                    nsteps=settings["nsteps"],
                    progress=settings["progress"])

            else:
                context = mp.get_context()
                with context.Pool(processes=settings["ncores"], initializer=_initialize_bayesian_worker, initargs=(self.spamms_fit,)) as pool:
                    sampler = emcee.EnsembleSampler( settings["nwalkers"], settings["ndim"], _worker_log_probability, pool=pool, backend=backend)
                    self._sample_with_progress(sampler=sampler, initial_positions=initial_positions, nsteps=settings["nsteps"], progress=settings["progress"])
            chain = sampler.get_chain()
            log_probability = sampler.get_log_prob()
            posterior_samples = sampler.get_chain(discard=settings["burnin"], thin=settings["thin"], flat=True)
            posterior_log_probability = sampler.get_log_prob( discard=settings["burnin"], thin=settings["thin"], flat=True)
            
            if posterior_samples.size == 0:
                raise RuntimeError("No posterior samples remain after burn-in and thinning.")

            credible_intervals = self._calculate_credible_intervals(posterior_samples)
            posterior_median_vector = credible_intervals["p50"].to_numpy(dtype=np.float64)
            maximum_index = int(np.argmax(posterior_log_probability))
            maximum_probability_vector = posterior_samples[maximum_index].copy()
            maximum_log_probability = float(posterior_log_probability[maximum_index])
            autocorrelation_time = self._estimate_autocorrelation_time(sampler)

            # One additional SPAMMS calculation creates the detailed
            # representative model at the posterior median.
            evaluation = self.spamms_fit.evaluate(posterior_median_vector)
            runtime = time.perf_counter() - run_start

            # Standard emcee ensemble updates evaluate each walker once
            # initially and once per MCMC step. The final representative
            # model adds one further SPAMMS calculation.
            estimated_evaluations = (settings["nwalkers"] * (settings["nsteps"] + 1) + 1)
            result = BayesianResult(
                chain=chain,
                log_probability=log_probability,
                posterior_samples=posterior_samples,
                acceptance_fraction=sampler.acceptance_fraction,
                credible_intervals=credible_intervals,
                posterior_median_vector=posterior_median_vector,
                maximum_probability_vector=maximum_probability_vector,
                maximum_log_probability=maximum_log_probability,
                autocorrelation_time=autocorrelation_time,
                best_parameters=evaluation["parameters"],
                free_parameter_names=(self.spamms_fit.parameters.free_names()),
                settings=self._serializable_settings(settings),
                runtime=runtime,
                n_evaluations=estimated_evaluations,
                evaluation=evaluation,
                backend_file=settings["backend_file"],
                metadata={"selected_lines": list(self.spamms_fit.spectrum.line_names), "evaluation_count_is_estimated": settings["ncores"] > 1},
                )

        except Exception as error:
            self._fail_run(error)
            raise
        return self._finish_run(result)

    def _serial_log_probability(self, theta: ArrayLike) -> float:
        """Calculate log probability without multiprocessing."""
        return _calculate_log_probability(theta=theta,spamms_fit=self.spamms_fit)

    @staticmethod
    def _sample_with_progress(sampler: emcee.EnsembleSampler,initial_positions: np.ndarray, nsteps: int, progress: bool) -> None:
        """Run emcee while displaying a tqdm step progress bar."""
        iterator = sampler.sample(initial_positions, iterations=nsteps, progress=False)
        if not progress:
            for _ in iterator:
                pass
            return
        with tqdm(total=nsteps, desc="Bayesian SPAMMS fit", unit="step") as progress_bar:
            for _ in iterator:
                progress_bar.update(1)

    def _prepare_settings(
        self,
        *,
        nsteps: int,
        nwalkers: int | None,
        ncores: int,
        initial_guess: ArrayLike | None,
        burnin: int | None,
        thin: int,
        seed: int | None,
        backend_file: str | Path | None,
        overwrite_backend: bool,
        progress: bool,
        show_recommendations: bool) -> dict[str, Any]:
        """Validate and normalize Bayesian settings."""

        ndim = self.spamms_fit.parameters.n_free
        minimum_walkers = 2 * ndim
        recommended_walkers = 4 * ndim
        if nwalkers is None:
            nwalkers = recommended_walkers
        nwalkers = int(nwalkers)
        nsteps = int(nsteps)
        ncores = int(ncores)
        thin = int(thin)

        if nsteps < 1:
            raise ValueError("nsteps must be at least 1.")
        if nwalkers < minimum_walkers:
            raise ValueError(f"emcee requires at least 2 * ndim = {minimum_walkers} walkers for the default ensemble move. Received {nwalkers}.")
        if nwalkers < recommended_walkers:
            warnings.warn(f"Using {nwalkers} walkers for {ndim} parameters. At least {recommended_walkers} walkers is recommended", RuntimeWarning, stacklevel=2)
        if ncores < 1:
            raise ValueError("ncores must be at least 1.")
        if ncores > nwalkers:
            warnings.warn(f"ncores={ncores} exceeds nwalkers={nwalkers}. Extra workers cannot evaluate additional walkers within one ensemble "
                f"update. Using ncores={nwalkers}.", RuntimeWarning, stacklevel=2)
            ncores = nwalkers
        if thin < 1:
            raise ValueError("thin must be at least 1.")
        if burnin is None:
            burnin = nsteps // 2
        burnin = int(burnin)
        if burnin < 0:
            raise ValueError("burnin cannot be negative.")
        if burnin >= nsteps:
            raise ValueError("burnin must be smaller than nsteps.")
        prepared_initial_guess = self._prepare_initial_guess(initial_guess)

        if seed is not None:
            seed = int(seed)
        if backend_file is None:
            backend_file = self.spamms_fit.config.results_directory / "chain.h5"
        else:
            backend_file = Path(backend_file).expanduser().resolve()

        return {
            "ndim": ndim,
            "nsteps": nsteps,
            "nwalkers": nwalkers,
            "ncores": ncores,
            "minimum_walkers": minimum_walkers,
            "recommended_walkers": recommended_walkers,
            "burnin": burnin,
            "thin": thin,
            "seed": seed,
            "initial_guess": prepared_initial_guess,
            "backend_file": backend_file,
            "overwrite_backend": bool(overwrite_backend),
            "progress": bool(progress),
            "show_recommendations": bool(show_recommendations),
            }

    def _prepare_initial_guess(self, initial_guess: ArrayLike | None) -> np.ndarray:
        """Prepare the central walker initialization point."""
        if initial_guess is None:
            guess = self.spamms_fit.parameters.initial_vector()
            if not self.spamms_fit.parameters.vector_in_bounds(guess):
                bounds = np.asarray(self.spamms_fit.parameters.continuous_bounds(),dtype=np.float64)
                guess = (bounds[:, 0] + bounds[:, 1]) / 2.0
            return guess
        guess = np.asarray(initial_guess, dtype=np.float64)

        if guess.ndim != 1:
            raise ValueError("initial_guess must be one-dimensional.")
        if guess.size != self.spamms_fit.parameters.n_free:
            raise ValueError( "initial_guess size does not match the number of free parameters.")
        if not self.spamms_fit.parameters.vector_in_bounds(guess):
            raise ValueError("initial_guess must lie strictly inside every parameter bound.")
        return guess

    def _initialize_walkers(
        self, rng: np.random.Generator, nwalkers: int, initial_guess: np.ndarray) -> np.ndarray:
        """
        Initialize walkers uniformly within the parameter bounds. The first walker is placed at the supplied initial guess.
        """
        bounds = np.asarray(self.spamms_fit.parameters.continuous_bounds(), dtype=np.float64)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        positions = rng.uniform(lower, upper, size=(nwalkers, self.spamms_fit.parameters.n_free))
        positions[0] = initial_guess
        return positions

    @staticmethod
    def _prepare_backend(
        *,
        backend_file: Path,
        nwalkers: int,
        ndim: int,
        overwrite: bool) -> emcee.backends.HDFBackend:

        """Create and initialize the emcee HDF5 backend."""
        backend_file.parent.mkdir(parents=True, exist_ok=True)
        if backend_file.exists() and not overwrite:
            raise FileExistsError(f"Bayesian backend already exists: {backend_file}. Set overwrite_backend=True to reset it.")
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        return backend

    def _calculate_credible_intervals(self,posterior_samples: np.ndarray ) -> pd.DataFrame:
        """Calculate 16th, 50th and 84th percentiles."""

        percentiles = np.percentile( posterior_samples, [16.0, 50.0, 84.0], axis=0)
        records = []
        for index, parameter_name in enumerate(self.spamms_fit.parameters.free_names()):
            p16 = float(percentiles[0, index])
            p50 = float(percentiles[1, index])
            p84 = float(percentiles[2, index])
            records.append({
                    "parameter": parameter_name,
                    "p16": p16,
                    "p50": p50,
                    "p84": p84,
                    "minus": p50 - p16,
                    "plus": p84 - p50
                    })
        return pd.DataFrame(records)

    @staticmethod
    def _estimate_autocorrelation_time(sampler: emcee.EnsembleSampler) -> np.ndarray | None:
        """Estimate autocorrelation times when the chain is long enough."""

        try:
            autocorrelation_time = sampler.get_autocorr_time(quiet=False)
        except emcee.autocorr.AutocorrError:
            return None
        if not np.all(np.isfinite(autocorrelation_time)):
            return None
        return np.asarray(autocorrelation_time, dtype=np.float64)

    def _prepare_subprocess_threads(self, ncores: int) -> None:
        """
        Prevent numerical-library oversubscription for parallel runs. Existing user-specified environment overrides are preserved.
        """

        if ncores <= 1:
            return
        environment = dict(self.spamms_fit.config.environment)
        thread_variables = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")
        for variable in thread_variables:
            environment.setdefault(variable, "1")
        self.spamms_fit.config.environment = environment

    @staticmethod
    def resource_recommendations(ndim: int, nwalkers: int | None = None) -> dict[str, int]:
        """Return minimum and recommended walker/core values."""
        ndim = int(ndim)
        if ndim < 1:
            raise ValueError("ndim must be at least 1.")
        minimum_walkers = 2 * ndim
        recommended_walkers = 4 * ndim
        if nwalkers is None:
            nwalkers = recommended_walkers
        available_cpus = os.cpu_count() or 1
        return {
            "minimum_walkers": minimum_walkers,
            "recommended_walkers": recommended_walkers,
            "minimum_cores": 1,
            "maximum_useful_cores": min(int(nwalkers), available_cpus)
            }

    def _print_resource_guidance(self,settings: Mapping[str, Any]) -> None:
        """Print sampler size and batching guidance."""

        nwalkers = int(settings["nwalkers"])
        ncores = int(settings["ncores"])
        batches_per_step = int(np.ceil(nwalkers / ncores))
        print()
        print("Bayesian SPAMMSFit configuration")
        print("================================")
        print(f"Free parameters: {settings['ndim']}")
        print("Parameter order: "
            + ", ".join(self.spamms_fit.parameters.free_names()))
        print(f"Walkers: {nwalkers}")
        print(f"Minimum walkers: {settings['minimum_walkers']}")
        print(f"Recommended walkers: {settings['recommended_walkers']}")
        print(f"Worker processes: {ncores}")
        print(f"Approximate SPAMMS batches per step: {batches_per_step}")
        print(f"MCMC steps: {settings['nsteps']}")
        print("Approximate likelihood evaluations:", f"{nwalkers * (settings['nsteps'] + 1)}")
        print("Selected lines: "
            + ", ".join(self.spamms_fit.spectrum.line_names))
        print()

    @staticmethod
    def _serializable_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        """Convert sampler settings to JSON-compatible values."""
        serializable = dict(settings)
        serializable["backend_file"] = str(serializable["backend_file"])
        initial_guess = serializable["initial_guess"]
        serializable["initial_guess"] = initial_guess.tolist()
        return serializable
        
    def __repr__(self) -> str:
        """Return a concise representation."""
        return (f"BayesianFit(status={self.status!r}, n_free={self.spamms_fit.parameters.n_free})")
