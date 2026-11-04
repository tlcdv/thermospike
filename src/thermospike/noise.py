"""Noise sources for stochastic neurons.

Every source counts the random bits it gives out, so a sampler can report
the cost of its randomness next to the quality of its samples.
"""

from typing import Optional, Protocol

import numpy as np


class NoiseSource(Protocol):
    """A source of uniform numbers on (0, 1) that counts its random bits."""

    bits_per_draw: int
    bits_used: int

    def uniform(self) -> float:
        """Return one uniform number on (0, 1)."""
        ...


class NumpyNoise:
    """Software noise from the NumPy PCG64 generator."""

    def __init__(self, seed: Optional[int] = None, bits_per_draw: int = 32):
        self._rng = np.random.default_rng(seed)
        self.bits_per_draw = bits_per_draw
        self.bits_used = 0

    def uniform(self) -> float:
        self.bits_used += self.bits_per_draw
        return float(self._rng.random())


# Galois mask for the maximal length polynomial x^32 + x^22 + x^2 + x + 1.
LFSR32_TAPS = 0x80200003


class LFSRNoise:
    """A 32 bit Galois linear feedback shift register, as a cheap FPGA would use.

    Each draw shifts the register `bits_per_draw` times and reads the output
    bits as an integer, so the resolution of one draw is 2 ** -bits_per_draw.
    """

    def __init__(self, seed: int = 1, bits_per_draw: int = 16):
        state = seed & 0xFFFFFFFF
        if state == 0:
            raise ValueError("an LFSR seed must be nonzero in its low 32 bits")
        if not 1 <= bits_per_draw <= 32:
            raise ValueError("bits_per_draw must be between 1 and 32")
        self.state = state
        self.bits_per_draw = bits_per_draw
        self.bits_used = 0

    def next_bit(self) -> int:
        bit = self.state & 1
        self.state >>= 1
        if bit:
            self.state ^= LFSR32_TAPS
        return bit

    def uniform(self) -> float:
        value = 0
        for _ in range(self.bits_per_draw):
            value = (value << 1) | self.next_bit()
        self.bits_used += self.bits_per_draw
        return (value + 0.5) / (1 << self.bits_per_draw)
