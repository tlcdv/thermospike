"""Small example problems written as Boltzmann machines."""

from typing import Iterable, Optional, Tuple

import numpy as np

from .model import BoltzmannMachine


def random_machine(n: int, scale: float = 1.0, seed: Optional[int] = None) -> BoltzmannMachine:
    """Dense random weights and biases, drawn from a normal distribution with this scale."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.normal(0.0, scale, (n, n)), k=1)
    return BoltzmannMachine(upper + upper.T, rng.normal(0.0, scale, n))


def and_gate(strength: float = 3.0) -> BoltzmannMachine:
    """Neurons (A, B, C) with low energy only where C = A AND B.

    The energy is strength * (3C + AB - 2AC - 2BC). It is 0 on the four rows of
    the truth table and at least `strength` on every other state. Clamp A and B
    to compute C, or clamp C to 1 to run the gate backwards and find A = B = 1.
    """
    s = strength
    weights = [[0.0, -s, 2 * s],
               [-s, 0.0, 2 * s],
               [2 * s, 2 * s, 0.0]]
    return BoltzmannMachine(weights, [0.0, 0.0, -3 * s])


def max_cut(n: int, edges: Iterable[Tuple[int, int]]) -> BoltzmannMachine:
    """Lowest energy states are the largest cuts of the graph. Energy = minus the cut size."""
    adjacency = np.zeros((n, n))
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1.0
    return BoltzmannMachine(-2 * adjacency, adjacency.sum(axis=1))


def cut_size(edges: Iterable[Tuple[int, int]], z: np.ndarray) -> int:
    return sum(int(z[i] != z[j]) for i, j in edges)
