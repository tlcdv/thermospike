"""Exact distributions by enumeration, to check samplers on small networks."""

from typing import Dict, Optional

import numpy as np

from .model import BoltzmannMachine

MAX_EXACT_NEURONS = 20


def all_states(n: int) -> np.ndarray:
    """Every binary state of n neurons. Row i is the state with index i."""
    if n > MAX_EXACT_NEURONS:
        raise ValueError(f"enumeration is limited to {MAX_EXACT_NEURONS} neurons")
    index = np.arange(2 ** n)
    return ((index[:, None] >> np.arange(n)) & 1).astype(np.int8)


def state_index(states: np.ndarray) -> np.ndarray:
    """Index of each state row, with neuron k as bit k."""
    states = np.atleast_2d(states)
    return (states.astype(np.int64) << np.arange(states.shape[1])).sum(axis=1)


def exact_distribution(model: BoltzmannMachine, beta: float = 1.0,
                       clamp: Optional[Dict[int, int]] = None) -> np.ndarray:
    """Probability of every state. States that break a clamp get probability 0."""
    states = all_states(model.n)
    energies = np.array([model.energy(z) for z in states])
    logits = -beta * energies
    for k, value in (clamp or {}).items():
        logits[states[:, k] != value] = -np.inf
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def empirical_distribution(samples: np.ndarray) -> np.ndarray:
    """Fraction of samples in every state."""
    counts = np.bincount(state_index(samples), minlength=2 ** samples.shape[1])
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. Use p for the samples and q for the exact distribution."""
    support = p > 0
    if np.any(q[support] == 0):
        return float("inf")
    return float(np.sum(p[support] * np.log(p[support] / q[support])))
