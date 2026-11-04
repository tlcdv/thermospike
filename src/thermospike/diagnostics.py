"""Mixing diagnostics. Consecutive samples of a Markov chain are correlated, so
a fair cost comparison divides the cost by the effective number of samples."""

from typing import List, Optional

import numpy as np

from .model import BoltzmannMachine


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a series at every lag, computed with an FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    return acf / acf[0]


def integrated_autocorrelation_time(x: np.ndarray) -> float:
    """1 + 2 * sum of autocorrelations, truncated by Geyer's initial positive sequence.

    It is 1 for independent samples. A value of 10 means ten correlated samples
    carry about as much information as one independent sample.
    """
    if np.var(x) == 0:
        raise ValueError("the series is constant, so its autocorrelation time is undefined")
    rho = autocorrelation(x)
    total = 0.0
    for m in range(len(rho) // 2):
        pair = rho[2 * m] + rho[2 * m + 1]
        if pair <= 0:
            break
        total += pair
    return max(1.0, 2.0 * total - 1.0)


def effective_sample_size(samples: np.ndarray, model: Optional[BoltzmannMachine] = None) -> float:
    """Worst case effective sample size over every neuron that changes, and over the energy if a model is given."""
    series: List[np.ndarray] = [samples[:, k] for k in range(samples.shape[1]) if np.var(samples[:, k]) > 0]
    if model is not None:
        energies = np.array([model.energy(z) for z in samples])
        if np.var(energies) > 0:
            series.append(energies)
    if not series:
        raise ValueError("no series in the samples changes, so nothing can be measured")
    worst = max(integrated_autocorrelation_time(s) for s in series)
    return len(samples) / worst
