# Local high-N benchmark for spdee Div(strain)

The sandbox Claude is running in caps out somewhere around N=32 on
ball-mode Div(strain) due to memory limits (~4-8 GB per process). With
your local hardware (768 GB, 72 threads) we can verify the asymptotic
behavior at N=48, 64, 96, 128 to confirm:

1. **Div(strain) rate-3.6** on the pure-ℓ=0 manufactured solution
   `v = (1-r²) ẑ`. Does it stay at 3.6, drift toward 4, or decay?
2. **Multi-ℓ origin issue.** The mixed `v_z = (1-r²) r² cos²θ` solution
   shows rate-2 at small N — does that persist or improve?
3. **Full pytest suite** confirms no regressions.

## Quick start

```bash
tar xzf spdee_local_bench.tar.gz
cd spdee
pip install -e .  # if not already installed
pip install numpy scipy sympy pytest  # if missing

# Quick convergence study (N=16,24,32,48): a few minutes.
python scratch/bench_div_strain_high_N.py

# Push to N=128 (uses ~240 GB at the largest step):
python scratch/bench_div_strain_high_N.py --big

# Full test suite (would be helpful to verify all 393 pass):
python -m pytest tests/ -q
```

## What I expect to see

### Pure ℓ=0 Div(strain), current state
| N | err |
|---|---|
| 16 | 9.5e-6 |
| 24 | 2.2e-6 |
| 32 | 7.7e-7 |
| 48 | ? (want to confirm) |
| 64 | ? |

Rate from 16→32 is 3.59. The question is whether it stays there, drifts
toward 4 (suggesting the gap is a finite-N effect), or decays toward 3
(suggesting a deeper rate-3 mechanism).

### Mixed ℓ=0+ℓ=2 Div(strain)
| N | err |
|---|---|
| 16 | 8.9e-5 |
| 24 | 3.8e-5 |
| 32 | 2.0e-5 |
| 48 | ? |

Rate is ~2.1 at small N. This is the multi-ℓ leakage at the origin that
the single-factor L'Hospital can't fix. I'd like to know if it stays
rate-2 (confirms the multi-ℓ origin treatment is needed) or improves
(perhaps the leakage decays faster than I think).

### Pytest

393 tests should pass; 1 skipped. I confirmed `tests/test_div_tensor_walker.py`
and `tests/test_lap_vector*.py` pass; the full suite takes ~4 minutes
locally and gets killed in the sandbox.

## Output to send back

`bench_results.json` and any pytest failures. Even a screenshot of the
table printed by the benchmark script is fine.
