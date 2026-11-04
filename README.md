# thermospike

Stochastic spiking neurons that sample from Boltzmann distributions. The library counts what each sample costs in random bits, synaptic events and multiply adds.

thermospike sits between two fields. Thermodynamic and probabilistic computing use physical noise to sample (p-bits, Gibbs samplers). Neuromorphic engineering uses spikes and event driven synapses. Neural sampling theory (Buesing, Bill, Nessler and Maass, 2011) shows that a network of stochastic spiking neurons with a refractory period samples from the same Boltzmann distribution as a p-bit network. thermospike implements both and measures the difference in cost.

## Status

Version 0.1.0. Small networks in pure Python and NumPy. Every sampler is checked against the exact distribution, found by enumerating every state. An FPGA reference design is planned and does not exist yet.

## Install

```
pip install thermospike
```

From source:

```
git clone https://github.com/tlcdv/thermospike
cd thermospike
pip install -e ".[test]"
pytest
```

## Example

```python
from thermospike import SpikingSampler, NumpyNoise, and_gate, empirical_distribution, exact_distribution, kl_divergence

model = and_gate()                     # C = A AND B, as an energy function
clamp = {2: 1}                         # fix the output C to 1 and run the gate backwards
sampler = SpikingSampler(model, NumpyNoise(seed=0), tau=4, clamp=clamp)
samples = sampler.sample(20000)

print(samples.mean(axis=0))            # A and B are near 1
print(kl_divergence(empirical_distribution(samples), exact_distribution(model, clamp=clamp)))
print(sampler.counters)                # random bits, spikes, synaptic events, multiply adds
```

## What is in the library

| Part | Contents |
|---|---|
| `BoltzmannMachine` | Binary neurons (0 silent, 1 active), symmetric weights, biases |
| `PBitSampler` | Sequential Gibbs sampling, one p-bit per neuron |
| `SpikingSampler` | Neural sampling with an absolute refractory period of `tau` steps. With `tau=1` it is the p-bit sampler |
| `NumpyNoise`, `LFSRNoise` | Software noise, and a 32 bit LFSR model of the noise a small FPGA would use |
| `Counters` | Updates, random draws, random bits, spikes, state changes, synaptic events, dense multiply adds |
| `exact_distribution`, `kl_divergence` | Checks against enumeration for up to 20 neurons |
| `and_gate`, `max_cut`, `random_machine` | Example problems |

## First measurement

`python examples/compare_cost.py --neurons 8 --sweeps 40000 --seeds 0 1 2` on one random 8 neuron machine (model seed 0) gave this table. KL is KL(empirical || exact) in nats. With 256 states and 40000 samples, finite sampling alone gives about 0.003, so every row sits near that floor.

| Sampler | KL mean | KL sd | Random bits per sweep | Synaptic events per sweep | Dense multiply adds per sweep |
|---|---|---|---|---|---|
| p-bit, NumPy noise | 0.00260 | 0.00034 | 256.0 | 7.87 | 56.0 |
| p-bit, 16 bit LFSR | 0.00290 | 0.00004 | 128.0 | 7.95 | 56.0 |
| spiking, tau 2 | 0.00293 | 0.00012 | 154.7 | 6.67 | 33.8 |
| spiking, tau 4 | 0.00282 | 0.00023 | 104.2 | 5.27 | 22.8 |
| spiking, tau 8 | 0.00345 | 0.00019 | 78.9 | 3.87 | 17.3 |

At equal sweeps, the spiking sampler with tau 4 used 2.5 times fewer random bits and 2.5 times fewer multiply adds than the p-bit sampler, with KL inside the sampling floor. This is one machine and one size, so read it as a hypothesis. Event counts are not joules. A fair energy claim needs a hardware cost per event and a measure of mixing speed (effective samples per sweep), which the library does not have yet.

## Plan

1. Effective sample size and autocorrelation, so samplers are compared per independent sample.
2. Larger sparse networks and a vectorised sampler.
3. An FPGA reference design (Tang Nano 9K) with a host script that checks the hardware samples against the exact distribution.
4. Comparisons with other libraries on the same problems.

Issues and pull requests are welcome.

## References

- L. Buesing, J. Bill, B. Nessler, W. Maass. Neural Dynamics as Sampling. PLoS Computational Biology, 2011.
- K. Y. Camsari, R. Faria, B. M. Sutton, S. Datta. Stochastic p-bits for invertible logic. Physical Review X, 2017.

## License

MIT
