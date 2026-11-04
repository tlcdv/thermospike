"""thermospike: stochastic spiking neurons that sample, with noise as the resource."""

from .exact import all_states, empirical_distribution, exact_distribution, kl_divergence, state_index
from .model import BoltzmannMachine
from .noise import LFSRNoise, NoiseSource, NumpyNoise
from .problems import and_gate, cut_size, max_cut, random_machine
from .samplers import Counters, PBitSampler, SpikingSampler

__version__ = "0.1.0"

__all__ = [
    "BoltzmannMachine",
    "Counters",
    "LFSRNoise",
    "NoiseSource",
    "NumpyNoise",
    "PBitSampler",
    "SpikingSampler",
    "all_states",
    "and_gate",
    "cut_size",
    "empirical_distribution",
    "exact_distribution",
    "kl_divergence",
    "max_cut",
    "random_machine",
    "state_index",
]
