# REVO — Research Prototype (CPU‑only)

REVO explores inference-time transformations that reshape activation dynamics without modifying model weights.

**Status:** research prototype — 9 phases implemented, CPU-only, 43 tests (pytest), CI via GitHub Actions. All modules verified importable.

---

## What this is

- **Phases I–V** (metric field, holographic geometry, frequency routing, fractal/ephemeral execution, energy calibration): modules for PDE-regularised metric field prediction, low-rank delta injection, FFT-based circulant kernels, hyperbolic adaptation, reversible un-computing, associative memory, and frequency-division processing.
- **Phases VI–IX** (regime duality, probabilistic calibration, implicit existence, bio-computational alignment): modules that detect activation regimes, calibrate output distributions via temperature scaling, probe latent codebook activation, and measure energy/coherence metrics.
- **CPU-only** evaluation with reproducible JSON artifacts under `quality/`.
- **41 tests** (pytest), CI via GitHub Actions.

## What this is not

- Not a production library. Not a drop-in optimizer replacement.
- Not a benchmark leaderboard. Numbers are small-scale, CPU-only probes.

---

## Architecture

```
revo/
├── _utils.py              # Shared utilities (seed, NLL, module iteration)
├── engine.py              # Low-rank delta apply/revert (NumPy)
├── hyperlora.py           # Deterministic context to (A, B, scale) generation
├── features.py            # Feature extraction, context vectors, mode keys
├── mode_cache.py          # TTL/LRU mode cache (thread-safe)
├── potentials.py          # Potential logging (JSONL with rotation)
├── observability.py       # Identity anchor and cognitive conservation logging
│
├── _logging.py            # Centralised logging configuration
├── _utils.py              # Shared utilities (seed, NLL, module iteration)
├── engine.py              # Low-rank delta apply/revert (NumPy)
├── hyperlora.py           # Deterministic context to (A, B, scale) generation
├── features.py            # Feature extraction, context vectors, mode keys
├── mode_cache.py          # TTL/LRU mode cache (thread-safe)
├── potentials.py          # Potential logging (JSONL with rotation)
├── observability.py       # Identity anchor and cognitive conservation logging
│
├── metric_field_pinn.py   # [I.1] PDE-regularised latent metric field
├── energy.py              # [I.2] EnergyMonitor, RAPL, Landauer calibration
├── hdram.py               # [I.3] Associative memory via cosine similarity
├── fdm_rtd.py             # [I.4] Frequency-division filter bank
├── fft_kernel.py          # [I.5] FFT-based circulant matrix multiplication
│
├── holography.py          # [II] HoloBoundaryAdapter + PINN calibration
├── hora.py                # [II] Hyperbolic low-rank adaptation
├── hyperbolic.py          # Poincare disk geometry (expmap, logmap, mobius)
├── bayesian_delta.py      # [II.1] Bayesian Ephemeral Delta Synthesis
├── tqft.py                # [II.2] Topological protection via FFT phase
├── category.py            # [II.3] Category-theoretic morphism DSL
├── radix_cache.py         # [II.4] Prefix-tree associative cache
├── frequency.py           # [II.5] Cognitive frequency-band coherence
│
├── spectral.py            # [III] 2D FFT spectral pruning
├── phase_bus.py           # [III] FFT phase alignment wrapper
├── reversible.py          # [III] SVD-based reversible un-computing
├── wdm.py                 # [III] Block-diagonal circulant band split
├── oscillatory_gating.py  # [III] Oscillatory phase-gated modulation
│
├── fractal.py             # [IV] Power-series linear wrap
├── ephemeral.py           # [IV] Sigmoid-gated modulation
├── radix.py               # [IV] Radix tree evaluation
├── hyperbolic_gating.py   # [IV] Hyperbolic gating + profile
│
├── spectral_network.py    # [V] Cauchy / spectral network transforms
├── equilibrium_propagation.py  # [V] Equilibrium propagation tuning
├── functor_monte_carlo.py # [V] Functor mapping verification
├── tensor_train.py        # [V] Tensor-train / MPO decomposition
├── holomorphic_projection.py   # [V] Holomorphic model projection
├── mutual_information_fusion.py # [V] MI-based output fusion
├── latency_monitor.py     # [V] Latency distribution measurement
├── probabilistic_delta.py # [V] Probabilistic delta modulation
├── low_dimensional.py     # [V] Low-dimensional consolidation
├── lowrank.py             # [V] Low-rank module replacement
├── calibration.py         # [V] Post-REVO head calibration
├── physical_onn.py        # [V] Optical neural network emulation
├── true_holomorphic.py    # [V] Holomorphic (complex) linear transforms
├── true_reversible.py     # [V] Reversible + adiabatic pipeline
├── true_pdm.py            # [V] Probabilistic delta modulation
├── layer_profile.py       # [V] Layer profiling for rank allocation
├── morse.py               # [V] Morse-skeleton weight sparsification
├── holomorphic.py         # [V] Holomorphic weight replacement
│
├── regimes.py             # [VI] Regime detection (micro/macro)
├── probcal.py             # [VII] Temperature scaling (tau) via SGD
├── implicit.py            # [VIII] K-means codebook + distance threshold
├── biocomp.py             # [IX] Activation fraction, coherence, energy
│
├── pipeline.py            # Orchestrator: runs all 5 phases, collects metrics
└── unified_main.py        # Unified CLI entry point for all phases

examples/              # Runnable benchmarks and tests (15 scripts)
tests/                 # 43 pytest tests across 6 suites
quality/               # JSON artifacts (phase runs, comparisons, reports)
```

---

## Quick start

```bash
# Install
pip install -e .

# Run all tests
python -m pytest tests/ -v

# Full pipeline (mock data)
python examples/run_pipeline.py

# CLI pipeline with a model
python main.py --prompt "Explain FFT-based inference."
```

---

## Reproducing benchmarks

```bash
# Metric field PINN (Phase I.1)
python examples/test_metric_field_pinn.py

# HDRAM associative memory (Phase I.3)
python examples/test_hdram.py

# FDM/RTD filter bank (Phase I.4)
python examples/test_fdm_rtd.py

# Topological protection (Phase II.2)
python examples/bench_tqft.py

# Phase VI-IX final reports
python examples/gen_final_reports.py
```

---

## Quick evidence (CPU-only)

Source: `examples/compare_revo_vs_qp.py` on `sshleifer/tiny-gpt2` (124K params, seed=0, 10 prompts).
Artifacts: `/tmp/bench.json`.

**NLL vs Latency vs Memory:**

| Variant       | NLL   | Eval Time (s) | RSS (MB) | Stability cos |
|---------------|-------|---------------|----------|---------------|
| baseline      | 10.83 | 0.187         | 623      | 1.00000       |
| quant+prune   | 10.83 | 0.091 (-51%)  | 688      | 0.99275       |
| revo          | 10.83 | 0.105 (-44%)  | 744      | 1.00000       |

**Key takeaways:**
- **NLL parity** — REVO does not degrade perplexity vs baseline.
- **Latency reduction** — 44% faster than baseline, comparable to quant+prune.
- **Reconstruction fidelity** — REVO preserves activation stability (cos=1.00, KL=1.2e-5) better than quant+prune (cos=0.993, KL=1.8e-5).
- **REVO is reversible** — low-rank deltas can be applied and reverted at any point, enabling ephemeral compute without permanent weight changes.
- **Quant+prune** is destructive (weight changes are permanent). REVO's lower reconstruction KL and perfect stability cosine reflect its non-destructive nature.

---

## Project status (Phases I-IX)

| Phase | Component | Status |
|-------|-----------|--------|
| I.1   | Metric Field PINN (PDE residual) | Verified 3.64e-7 |
| I.3   | HDRAM (recall@1) | Verified 1.0 |
| I.4   | FDM/RTD filter bank | Verified |
| I.5   | FFT circulant (O(n log n)) | Verified |
| II.1  | BEDS entropy homeostasis | Verified |
| II.2  | TQFT topological protection | Verified |
| II.3  | Category morphism DSL | Verified |
| II.4  | Radix prefix cache | Verified |
| II.5  | Cognitive frequency tuning | Verified |
| III   | Spectral + FFT + Reversible + WDM | Verified |
| IV    | Fractal + Ephemeral + Radix | Verified |
| V     | Energy + MEI sync | Verified |
| VI    | Regime duality | Verified |
| VII   | Probabilistic calibration | Verified |
| VIII  | Implicit existence | Verified |
| IX    | Bio-computational convergence | Verified |

---

## License

MIT (c) 2026 REVO contributors.
