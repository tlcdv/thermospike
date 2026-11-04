"""Boltzmann machines over binary neurons.

A state z has one entry per neuron, 1 for active (spiking) and 0 for silent.
The energy is E(z) = -1/2 z.W.z - b.z and the target distribution is
p(z) proportional to exp(-beta * E(z)).
"""

from typing import Sequence

import numpy as np


class BoltzmannMachine:
    """Symmetric weights with a zero diagonal, and one bias per neuron."""

    def __init__(self, weights: Sequence[Sequence[float]], biases: Sequence[float]):
        w = np.asarray(weights, dtype=float)
        b = np.asarray(biases, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError("weights must be a square matrix")
        if b.shape != (w.shape[0],):
            raise ValueError("biases must have one entry per neuron")
        if not np.allclose(w, w.T):
            raise ValueError("weights must be symmetric")
        if np.any(np.diag(w) != 0):
            raise ValueError("weights must have a zero diagonal")
        self.weights = w
        self.biases = b
        # Number of synapses that leave each neuron. An event driven chip pays
        # this many synaptic updates each time the neuron changes state.
        self.fan_out = np.count_nonzero(w, axis=0)

    @property
    def n(self) -> int:
        return len(self.biases)

    def energy(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return float(-0.5 * z @ self.weights @ z - self.biases @ z)

    def potential(self, z: np.ndarray, k: int) -> float:
        """Membrane potential of neuron k, which is minus the energy change of switching it on."""
        return float(self.biases[k] + self.weights[k] @ z)
