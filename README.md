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
| `effective_sample_size`, `integrated_autocorrelation_time` | Mixing diagnostics, so costs are compared per independent sample |
| `and_gate`, `max_cut`, `random_machine` | Example problems |

## First measurement

`python examples/compare_cost.py --neurons 8 --sweeps 40000 --seeds 0 1 2` on one random 8 neuron machine (model seed 0, weight scale 1) gave this table. KL is KL(empirical || exact) in nats. With 256 states and 40000 samples, finite sampling alone gives about 0.003, so every row sits near that floor. ESS is the effective sample size, the worst case over every neuron and the energy. Costs are per effective sample.

| Sampler | KL mean | ESS mean | Random bits per ESS | Synaptic events per ESS | Dense multiply adds per ESS |
|---|---|---|---|---|---|
| p-bit, NumPy noise (32 bit draws) | 0.00260 | 27909 | 367.1 | 11.28 | 80.3 |
| p-bit, 16 bit LFSR | 0.00290 | 29014 | 176.5 | 10.96 | 77.2 |
| spiking, tau 2 | 0.00293 | 18400 | 336.4 | 14.51 | 73.6 |
| spiking, tau 4 | 0.00282 | 9675 | 430.9 | 21.80 | 94.3 |
| spiking, tau 8 | 0.00345 | 4902 | 644.5 | 31.64 | 141.0 |

Per sweep, the spiking sampler draws fewer random numbers, because a refractory neuron draws none. Per effective sample, this saving disappears. The refractory period makes consecutive samples more correlated, and at tau 4 and 8 the spiking sampler costs more than the p-bit sampler on every count. Only tau 2 is close, with 8 percent fewer random bits and multiply adds and 29 percent more synaptic events. The largest saving in this table comes from the noise source. A 16 bit LFSR halves the random bits with no measured loss of quality.

This is one small machine, so these are hypotheses. Event counts are not joules. An energy claim needs a hardware cost for each kind of event.

## Plan

1. Larger sparse networks, more machines per size, and a vectorised sampler.
2. Spiking variants that keep the per sweep saving without the extra correlation.
3. An FPGA reference design (Tang Nano 9K) with a host script that checks the hardware samples against the exact distribution.
4. Comparisons with other libraries on the same problems.

Issues and pull requests are welcome.

## References

- L. Buesing, J. Bill, B. Nessler, W. Maass. Neural Dynamics as Sampling. PLoS Computational Biology, 2011.
- K. Y. Camsari, R. Faria, B. M. Sutton, S. Datta. Stochastic p-bits for invertible logic. Physical Review X, 2017.

## License

MIT
