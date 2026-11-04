"""Compare sample quality and event cost for p-bit and spiking samplers.

Every sampler runs the same number of sweeps on the same random machine.
Quality is KL(empirical || exact) in nats. Consecutive samples are correlated,
so cost is reported per effective sample (worst case over neurons and energy).

    python examples/compare_cost.py --neurons 8 --sweeps 40000 --seeds 0 1 2
"""

import argparse

import numpy as np

from thermospike import (PBitSampler, SpikingSampler, NumpyNoise, LFSRNoise, effective_sample_size,
                         empirical_distribution, exact_distribution, kl_divergence, random_machine)

BURN_IN = 100


def run(kind, tau, model, seed, sweeps):
    noise = LFSRNoise(seed=seed + 1) if kind == "lfsr" else NumpyNoise(seed=seed)
    sampler = PBitSampler(model, noise) if tau is None else SpikingSampler(model, noise, tau=tau)
    samples = sampler.sample(sweeps, burn_in=BURN_IN)
    kl = kl_divergence(empirical_distribution(samples), exact_distribution(model))
    ess = effective_sample_size(samples, model)
    c = sampler.counters
    # Cost of the kept sweeps only, scaled from the totals that include burn in.
    kept = sweeps / (sweeps + BURN_IN)
    return (kl, ess, c.random_bits * kept / ess, c.synaptic_events * kept / ess,
            c.dense_multiply_adds * kept / ess)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=40000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0, help="standard deviation of weights and biases")
    args = parser.parse_args()

    model = random_machine(args.neurons, scale=args.scale, seed=args.model_seed)
    configs = [("p-bit, numpy", "numpy", None), ("p-bit, lfsr16", "lfsr", None)]
    configs += [(f"spiking tau={t}, numpy", "numpy", t) for t in (2, 4, 8)]

    print(f"neurons={args.neurons} sweeps={args.sweeps} seeds={args.seeds} "
          f"model_seed={args.model_seed} scale={args.scale}")
    print(f"{'sampler':24s} {'KL mean':>9s} {'KL sd':>8s} {'ESS mean':>9s} {'ESS sd':>7s} "
          f"{'bits/ESS':>9s} {'syn ev/ESS':>11s} {'MAC/ESS':>8s}")
    for name, kind, tau in configs:
        rows = np.array([run(kind, tau, model, s, args.sweeps) for s in args.seeds])
        kl, ess, bits, syn, mac = rows.mean(axis=0)
        print(f"{name:24s} {kl:9.5f} {rows[:, 0].std():8.5f} {ess:9.0f} {rows[:, 1].std():7.0f} "
              f"{bits:9.1f} {syn:11.2f} {mac:8.1f}")


if __name__ == "__main__":
    main()
