import numpy as np
import pytest

from thermospike import (
    BoltzmannMachine,
    LFSRNoise,
    NumpyNoise,
    PBitSampler,
    SpikingSampler,
    all_states,
    and_gate,
    cut_size,
    effective_sample_size,
    empirical_distribution,
    exact_distribution,
    integrated_autocorrelation_time,
    kl_divergence,
    max_cut,
    random_machine,
)

# KL(empirical || exact) for 32 states from 40000 correlated samples sits near
# 0.001 nats from finite sampling alone. A wrong sampler lands far above 0.01.
KL_LIMIT = 0.01


# ------------------------------------------------------------------ noise


def test_lfsr_is_deterministic_and_counts_bits():
    a, b = LFSRNoise(seed=7, bits_per_draw=16), LFSRNoise(seed=7, bits_per_draw=16)
    assert [a.uniform() for _ in range(50)] == [b.uniform() for _ in range(50)]
    assert a.bits_used == 50 * 16


def test_lfsr_output_is_close_to_uniform():
    noise = LFSRNoise(seed=12345)
    values = np.array([noise.uniform() for _ in range(20000)])
    assert 0 < values.min() and values.max() < 1
    assert abs(values.mean() - 0.5) < 0.01
    counts, _ = np.histogram(values, bins=10, range=(0, 1))
    assert counts.min() > 1800 and counts.max() < 2200


def test_lfsr_rejects_zero_seed():
    with pytest.raises(ValueError):
        LFSRNoise(seed=0)


# ------------------------------------------------------------------ model


def test_model_rejects_asymmetric_weights():
    with pytest.raises(ValueError):
        BoltzmannMachine([[0, 1], [2, 0]], [0, 0])


def test_energy_matches_hand_calculation():
    model = BoltzmannMachine([[0, 2], [2, 0]], [1, -1])
    # E = -W01 z0 z1 - b.z
    assert model.energy([1, 1]) == pytest.approx(-2 - 1 + 1)
    assert model.energy([1, 0]) == pytest.approx(-1)
    assert model.potential(np.array([0.0, 1.0]), 0) == pytest.approx(1 + 2)


def test_exact_distribution_sums_to_one_and_respects_clamp():
    model = random_machine(4, seed=0)
    p = exact_distribution(model, clamp={2: 1})
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p[all_states(4)[:, 2] == 0] == 0)


# ------------------------------------------------------------------ samplers against exact enumeration


@pytest.mark.parametrize("noise", [NumpyNoise(seed=1), LFSRNoise(seed=99)], ids=["numpy", "lfsr"])
def test_pbit_sampler_matches_exact_distribution(noise):
    model = random_machine(5, seed=3)
    samples = PBitSampler(model, noise).sample(40000)
    kl = kl_divergence(empirical_distribution(samples), exact_distribution(model))
    assert kl < KL_LIMIT


@pytest.mark.parametrize("tau", [2, 4])
def test_spiking_sampler_matches_exact_distribution(tau):
    model = random_machine(5, seed=3)
    samples = SpikingSampler(model, NumpyNoise(seed=2), tau=tau).sample(40000)
    kl = kl_divergence(empirical_distribution(samples), exact_distribution(model))
    assert kl < KL_LIMIT


def test_spiking_sampler_with_tau_one_is_the_pbit_sampler():
    model = random_machine(4, seed=5)
    pbit = PBitSampler(model, NumpyNoise(seed=8)).sample(2000, burn_in=0)
    spiking = SpikingSampler(model, NumpyNoise(seed=8), tau=1).sample(2000, burn_in=0)
    assert np.array_equal(pbit, spiking)


def test_spiking_sampler_skips_random_draws_while_refractory():
    model = random_machine(5, seed=3)
    pbit = PBitSampler(model, NumpyNoise(seed=1))
    spiking = SpikingSampler(model, NumpyNoise(seed=1), tau=4)
    pbit.sample(5000)
    spiking.sample(5000)
    assert pbit.counters.random_draws == pbit.counters.updates
    assert spiking.counters.random_draws < spiking.counters.updates
    assert spiking.counters.random_bits == 32 * spiking.counters.random_draws


def test_synaptic_events_follow_state_changes():
    model = random_machine(5, seed=3)
    sampler = PBitSampler(model, NumpyNoise(seed=4))
    sampler.sample(1000)
    # Dense random weights: every neuron reaches the other four.
    assert sampler.counters.synaptic_events == 4 * sampler.counters.state_changes
    assert sampler.counters.dense_multiply_adds == 4 * sampler.counters.updates


# ------------------------------------------------------------------ mixing diagnostics


def test_autocorrelation_time_of_independent_samples_is_one():
    x = np.random.default_rng(0).normal(size=20000)
    assert integrated_autocorrelation_time(x) == pytest.approx(1.0, abs=0.1)


def test_autocorrelation_time_of_ar1_series_matches_theory():
    # For x[t] = phi * x[t-1] + noise the time is (1 + phi) / (1 - phi) = 19 at phi = 0.9.
    rng = np.random.default_rng(1)
    x = np.zeros(200000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    assert integrated_autocorrelation_time(x) == pytest.approx(19.0, rel=0.15)


def test_constant_series_is_rejected():
    with pytest.raises(ValueError):
        integrated_autocorrelation_time(np.ones(100))


def test_effective_sample_size_skips_clamped_neurons():
    model = and_gate()
    samples = PBitSampler(model, NumpyNoise(seed=3), clamp={2: 1}).sample(5000)
    ess = effective_sample_size(samples, model)
    assert 0 < ess <= 5000


# ------------------------------------------------------------------ example problems


def test_and_gate_forward_and_backward():
    model = and_gate(strength=3.0)
    for a in (0, 1):
        for b in (0, 1):
            samples = PBitSampler(model, NumpyNoise(seed=a * 2 + b), clamp={0: a, 1: b}).sample(5000)
            assert samples[:, 2].mean() == pytest.approx(a & b, abs=0.1)
    clamp = {2: 1}
    samples = SpikingSampler(model, NumpyNoise(seed=11), tau=3, clamp=clamp).sample(20000)
    kl = kl_divergence(empirical_distribution(samples), exact_distribution(model, clamp=clamp))
    assert kl < KL_LIMIT
    assert np.mean(np.all(samples == 1, axis=1)) > 0.85


def test_annealing_finds_the_maximum_cut_of_a_ring():
    edges = [(i, (i + 1) % 6) for i in range(6)]
    sampler = PBitSampler(max_cut(6, edges), NumpyNoise(seed=0))
    for beta in np.linspace(0.1, 5.0, 200):
        sampler.beta = beta
        sampler.sweep()
    assert cut_size(edges, sampler.z) == 6
