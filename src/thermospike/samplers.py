"""Samplers that run a Boltzmann machine with stochastic neurons.

PBitSampler is sequential Gibbs sampling. Each neuron is a p-bit that turns on
with probability sigmoid(beta * u), where u is its membrane potential.

SpikingSampler is neural sampling with an absolute refractory period
(Buesing, Bill, Nessler and Maass, PLoS Comput Biol 2011). A neuron that
spikes stays active for tau steps. Outside that period it spikes with
probability sigmoid(beta * u - log(tau)). The active neurons are samples from
the same Boltzmann distribution, and the neuron draws no random bits while
it is refractory. With tau = 1 it is the p-bit sampler.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .model import BoltzmannMachine
from .noise import NoiseSource


@dataclass
class Counters:
    """Work done by a sampler. These are event counts, not joules."""

    updates: int = 0            # neuron update steps
    random_draws: int = 0       # uniform numbers drawn
    random_bits: int = 0        # bits behind those draws
    spikes: int = 0             # p-bit: off to on events. Spiking: every spike, also one that extends an active period
    state_changes: int = 0      # on to off and off to on events
    synaptic_events: int = 0    # fan out of every state change (event driven cost)
    dense_multiply_adds: int = 0  # cost of recomputing every potential from scratch


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class _Sampler:
    def __init__(self, model: BoltzmannMachine, noise: NoiseSource, beta: float = 1.0,
                 clamp: Optional[Dict[int, int]] = None, initial: Optional[np.ndarray] = None):
        self.model = model
        self.noise = noise
        self.beta = beta
        self.clamp = dict(clamp or {})
        self.z = np.zeros(model.n) if initial is None else np.asarray(initial, dtype=float).copy()
        for k, value in self.clamp.items():
            self.z[k] = value
        self.free = [k for k in range(model.n) if k not in self.clamp]
        self.counters = Counters()

    def _draw(self) -> float:
        bits_before = self.noise.bits_used
        value = self.noise.uniform()
        self.counters.random_draws += 1
        self.counters.random_bits += self.noise.bits_used - bits_before
        return value

    def _set(self, k: int, value: int) -> None:
        if self.z[k] != value:
            self.counters.state_changes += 1
            self.counters.synaptic_events += int(self.model.fan_out[k])
            self.z[k] = value

    def _potential(self, k: int) -> float:
        self.counters.dense_multiply_adds += self.model.n - 1
        return self.model.potential(self.z, k)

    def _update(self, k: int) -> None:
        raise NotImplementedError

    def sweep(self) -> None:
        """Update every free neuron once, in index order."""
        for k in self.free:
            self.counters.updates += 1
            self._update(k)

    def sample(self, n_samples: int, burn_in: int = 100) -> np.ndarray:
        """Run burn_in sweeps, then return the state after each of n_samples sweeps."""
        for _ in range(burn_in):
            self.sweep()
        out = np.empty((n_samples, self.model.n), dtype=np.int8)
        for i in range(n_samples):
            self.sweep()
            out[i] = self.z
        return out


class PBitSampler(_Sampler):
    """Sequential Gibbs sampling with p-bit neurons."""

    def _update(self, k: int) -> None:
        p_on = sigmoid(self.beta * self._potential(k))
        value = 1 if self._draw() < p_on else 0
        if value == 1 and self.z[k] == 0:
            self.counters.spikes += 1
        self._set(k, value)


class SpikingSampler(_Sampler):
    """Neural sampling with stochastic spiking neurons and a refractory period of tau steps."""

    def __init__(self, model: BoltzmannMachine, noise: NoiseSource, tau: int = 1, **kwargs):
        if tau < 1:
            raise ValueError("tau must be at least 1")
        super().__init__(model, noise, **kwargs)
        self.tau = tau
        self.log_tau = math.log(tau)
        # Steps left in the active period. A neuron is active while this is 1 or more.
        self.refractory = np.where(self.z == 1, 1, 0)

    def _update(self, k: int) -> None:
        if self.refractory[k] > 1:
            self.refractory[k] -= 1
            return
        p_spike = sigmoid(self.beta * self._potential(k) - self.log_tau)
        if self._draw() < p_spike:
            self.refractory[k] = self.tau
            self.counters.spikes += 1
            self._set(k, 1)
        else:
            self.refractory[k] = 0
            self._set(k, 0)
