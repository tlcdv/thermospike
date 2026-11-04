"""Compare sample quality and event cost for p-bit and spiking samplers.

Every sampler runs the same number of sweeps on the same random machine.
Quality is KL(empirical || exact) in nats. Cost is in event counts per sweep.

    python examples/compare_cost.py --neurons 8 --sweeps 40000 --seeds 0 1 2
"""

import argparse

import numpy as np

from thermospike import (PBitSampler, SpikingSampler, NumpyNoise, LFSRNoise, empirical_distribution,
                         exact_distribution, kl_divergence, random_machine)


def run(kind, tau, model, seed, sweeps):
    noise = LFSRNoise(seed=seed + 1) if kind == "lfsr" else NumpyNoise(seed=seed)
    sampler = PBitSampler(model, noise) if tau is None else SpikingSampler(model, noise, tau=tau)
    samples = sampler.sample(sweeps)
    kl = kl_divergence(empirical_distribution(samples), exact_distribution(model))
    c = sampler.counters
    per = 1.0 / (sweeps + 100)  # sample() also runs 100 burn in sweeps
    return kl, c.random_bits * per, c.synaptic_events * per, c.dense_multiply_adds * per


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=40000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()

    model = random_machine(args.neurons, seed=args.model_seed)
    configs = [("p-bit, numpy", "numpy", None), ("p-bit, lfsr16", "lfsr", None)]
    configs += [(f"spiking tau={t}, numpy", "numpy", t) for t in (2, 4, 8)]

    print(f"neurons={args.neurons} sweeps={args.sweeps} seeds={args.seeds} model_seed={args.model_seed}")
    print(f"{'sampler':24s} {'KL mean':>9s} {'KL sd':>8s} {'bits/sweep':>11s} {'syn ev/sweep':>13s} {'dense MAC/sweep':>16s}")
    for name, kind, tau in configs:
        rows = np.array([run(kind, tau, model, s, args.sweeps) for s in args.seeds])
        kl, bits, syn, mac = rows.mean(axis=0)
        print(f"{name:24s} {kl:9.5f} {rows[:, 0].std():8.5f} {bits:11.1f} {syn:13.2f} {mac:16.1f}")


if __name__ == "__main__":
    main()
