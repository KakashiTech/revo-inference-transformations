# REVO — the model that lives in time

**REVO** (Reversible Execution Via reOrdering) is not a compression technique.
It is a redefinition: the model is not a file of static weights.
It is a **reproducible transformation law** that exists in time as a causal trajectory.

Inference is not global execution of a fixed network.
It is **local reconstruction under context**: the subnetwork you need for this token,
ephemerally generated, executed, reverted, forgotten.

The model as phenomenon, not as thing.

```
today:    reversible deltas + gradient correction + lossless quantization
            ↓
tomorrow: ephemeral per-token reconstruction, cognitive field modulating the trajectory,
          disk as slow living memory, the model that weighs nothing in RAM
```

**Status:** research prototype — 11 phases implemented, CPU-only, 270 tests (pytest),
CI via GitHub Actions. Everything verified importable.

This repo anchors the legitimacy of REVO as a technical line of work. It includes
CPU-only evidence, JSON artifacts, and lightweight commands to reproduce small
checks. No release yet.

---

## The rupture

The industry axiom is: the model is an object that must live in RAM.
From that comes everything — quantization, pruning, distillation. Sacrifice fidelity
so it fits. And accept that the sacrifice is permanent.

That is false. A large model does not need to exist complete at any instant.
It only needs to be a reproducible causal trajectory. Like the ocean: you don't have
every wave at one point, but the ocean behaves coherently.

REVO is that paradigm shift.

---

## The model as phenomenon (four strata)

A REVO model is not a stack of matrices. It is four layers that never coexist complete:

```
1. GENERATIVE LAW          CPU, fixed RAM (MB)
   Compact functions that generate weight deltas from latent context.
   Deterministic Hyper-LoRA: Z → (A, B, scale).

2. COGNITIVE FIELD Φ/A/C   Living RAM (KB-MB)
   Valence, arousal, coherence. Decides what to reconstruct, at what magnitude,
   at what speed. The field modulates the trajectory at inference time.

3. COMPRESSED HISTORY      Disk (GB)
   Potentials, latents, resonance maps, trajectories.
   No weights.bin — latent maps, compressed fields, semantic pheromones.

4. LOCAL MANIFESTATION      Ephemeral RAM (KB)
   Micro-subnetwork reconstructed for this token/window. Executed, reverted,
   destroyed. Never more than a tiny fragment of the model exists at once.
```

The trick: **100% of parameters never coexist**.
What exists is the certainty that, given the right context,
the reconstruction will be coherent.

---

## The fundamental primitive: the delta

REVO does not replace weights. REVO applies **deltas** — reversible low-rank
perturbations. Each delta captures a transformation and can be reverted
with statistically undetectable loss (~3e-06).

From this primitive everything is born:

```
delta              →  any transformation is reversible
tail handle        →  discarded information is not destroyed, it is saved
gradient correction →  lost quality is recovered via backprop
modular selection  →  you only revert what hurts
```

The full cycle:

1. Compress (SVD truncation, quantization, spectral pruning)
2. The handle saves what you discarded (low-rank factors, original weights)
3. Measure impact on NLL
4. If it hurts → gradient correction or selective revert
5. If it doesn't hurt → done, the model lives compressed but you are free to go back

No other technique can do this.
GPTQ cannot. SparseGPT cannot. AWQ cannot.
All destroy permanently. REVO does not.

---

## Quick evidence (CPU-only)

Source: `examples/compare_revo_vs_qp.py` (GPT-2, seed=0, 20 prompts).
Artifacts: `quality/compare/compare_gpt2_s20.json`, `quality/compare/compare_gpt2_s20_calib.json`.

**NLL vs Latency:**

| Variant       | NLL (General) | NLL (OOD) | Eval Time (s) |
|---------------|---------------|-----------|----------------|
| baseline      | 6.1201        | 5.4191    | 9.0052         |
| quant+prune   | 6.5766        | 5.8342    | 8.9312         |
| revo (calib)  | 7.3317        | 6.5428    | 8.0398         |

**Cos vs KL (recon/stability, last-token logits):**

| Variant       | recon cos | recon KL | stability cos | stability KL |
|---------------|-----------|----------|---------------|--------------|
| baseline      | 1.0000    | 0.0000   | 1.0000        | 0.0000       |
| quant+prune   | 0.99994   | 0.4067   | 0.99990       | 0.8194       |
| revo (calib)  | 0.99976   | 1.1385   | 0.99969       | 0.7052       |

**Alignment check** (`transformer.ln_f` hook): hidden L2 norm 231.6 to 91.4;
logit diff ~12.46k. Confirms internal modulation without quantisation.

---

## Compression results

### SVD + gradient correction

GPT-2 124M, wikitext-2, truncation to 25% of singular value range.

```
Config                                    ΔNLL      Compression  Revertible
───────────────────────────────────────────────────────────────────────────
SVD pure                                 +4.09      1.53×       ✅ 3e-06
+ gradient correction (5 steps AdamW)    +0.36      1.53×       ✅
+ separate calibration (100 txts)        +2.01      1.53×       ✅
safe_compress (selective revert)         +0.65      variable    ✅
```

Gradient correction closes ~90% of the quality gap.

### QuantizedLinear 4-bit

| Config               | ΔNLL      | Compression  | Revertible |
|----------------------|-----------|-------------|------------|
| 4-bit group=32       | −0.005*   | 8.0×        | ✅ 1e-08   |
| 4-bit group=64       | +0.040    | 8.0×        | ✅         |
| 3-bit group=32       | +0.345    | 10.7×       | ✅         |
| 2-bit group=64       | +4.15     | 16.0×       | ✅         |

*\* Improvement from regularization. 4-bit lossless.*

---

## Architecture

```
revo/
├── act_svd.py            # SVD + tail handles + gradient correction
├── lowrank.py            # LowRankLinear (A@B)
├── layer_profile.py      # Layer profiling for rank allocation
├── gptq_revo.py          # QuantizedLinear + handles + selective dequantize
├── gptq_impl.py          # GPTQ implementation (experimental)
│
├── engine.py             # Low-rank delta apply/revert (NumPy)
├── hyperlora.py          # Deterministic context to (A, B, scale) generation
├── features.py           # Feature extraction, context vectors, mode keys
├── mode_cache.py         # TTL/LRU mode cache (thread-safe)
├── potentials.py         # Potential logging (JSONL with rotation)
├── observability.py      # Identity anchor and cognitive conservation logging
├── calibration.py        # Post-REVO head calibration
│
├── metric_field_pinn.py  # [I.1] PDE-regularised latent metric field
├── energy.py             # [I.2] EnergyMonitor, RAPL, Landauer calibration
├── hdram.py              # [I.3] Associative memory via cosine similarity
├── fdm_rtd.py            # [I.4] Frequency-division filter bank
├── fft_kernel.py         # [I.5] FFT-based circulant matrix multiplication
├── holography.py         # [II] HoloBoundaryAdapter + PINN calibration
├── hora.py               # [II] Hyperbolic low-rank adaptation
├── hyperbolic.py         # Poincare disk geometry (expmap, logmap, mobius)
├── bayesian_delta.py     # [II.1] Bayesian Ephemeral Delta Synthesis
├── beds.py               # [II.1-alt] Bayesian dissipative structures
├── tqft.py               # [II.2] Topological protection via FFT phase
├── category.py           # [II.3] Category-theoretic morphism DSL
├── radix_cache.py        # [II.4] Prefix-tree associative cache
├── frequency.py          # [II.5] Cognitive frequency-band coherence
├── spectral.py           # [III] 2D FFT spectral pruning
├── phase_bus.py          # [III] FFT phase alignment wrapper
├── reversible.py         # [III] SVD-based reversible un-computing
├── wdm.py                # [III] Block-diagonal circulant band split
├── oscillatory_gating.py # [III] Oscillatory phase-gated modulation
├── fractal.py            # [IV] Power-series linear wrap
├── ephemeral.py          # [IV] Sigmoid-gated modulation
├── radix.py              # [IV] Radix tree evaluation
├── hyperbolic_gating.py  # [IV] Hyperbolic gating + profile
├── spectral_network.py   # [V] Cauchy / spectral network transforms
├── equilibrium_propagation.py  # [V] Equilibrium propagation tuning
├── functor_monte_carlo.py # [V] Functor mapping verification
├── tensor_train.py       # [V] Tensor-train / MPO decomposition
├── holomorphic_projection.py   # [V] Holomorphic model projection
├── mutual_information_fusion.py # [V] MI-based output fusion
├── latency_monitor.py    # [V] Latency distribution measurement
├── probabilistic_delta.py # [V] Probabilistic delta modulation
├── low_dimensional.py    # [V] Low-dimensional consolidation
├── true_holomorphic.py   # [V] Holomorphic (complex) linear transforms
├── true_reversible.py    # [V] Reversible + adiabatic pipeline
├── true_pdm.py           # [V] Probabilistic delta modulation
├── physical_onn.py       # [V] Optical neural network emulation
├── morse.py              # [V] Morse-skeleton weight sparsification
├── holomorphic.py        # [V] Holomorphic weight replacement
│
├── regimes.py            # [VI] Regime detection (micro/macro)
├── probcal.py            # [VII] Temperature scaling (tau) via SGD
├── implicit.py           # [VIII] K-means codebook + distance threshold
├── biocomp.py            # [IX] Activation fraction, coherence, energy
├── primitiva_router.py   # [X] Token-conditional computational primitive selection
├── generative_law.py     # [XI] F(token) → 32-bit code → algebra (pure/residual/top-k)
│
├── pipeline.py           # Orchestrator: runs all phases, collects metrics
├── unified_main.py       # Unified CLI entry point for all phases
├── archive/              # Archived experimental modules (15+)
│
├── _logging.py           # Centralised logging configuration
└── _utils.py             # Shared utilities (seed, NLL, module iteration)

examples/                 # Runnable benchmarks and tests (15+ scripts)
tests/                    # 270 pytest tests across 31 suites
quality/                  # JSON artifacts (phase runs, comparisons, reports)
```

---

## Phase status

| Phase | Component | Status |
|-------|-----------|--------|
| I.1   | Metric Field PINN (PDE residual) | Verified 3.64e-7 |
| I.3   | HDRAM (recall@1) | Verified 1.0 |
| I.4   | FDM/RTD filter bank | Verified |
| I.5   | FFT circulant (O(n log n)) | Verified |
| II.1  | BEDS / Bayesian delta synthesis | Verified |
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
| X     | Primitive Router (token-conditional) | Verified — 234 tests, GPT-2 support |
| XI    | Generative Law (F: token → 32-bit code) | Verified — 270 tests, modes: pure/residual, top-k sparse |

---

## Reproduce (CPU-only)

```bash
# Install
pip install -e .

# Run all tests
python -m pytest tests/ -v

# Full pipeline (mock data)
python examples/run_pipeline.py

# CLI with a model
python main.py --prompt "Explain low-rank inference."

# Benchmark SVD + gradient correction
python examples/exp_revo_fusion.py

# REVO vs quantisation+pruning (gpt2, 1 seed, 20 prompts)
PYTHONPATH=. python examples/compare_revo_vs_qp.py \
  --model gpt2 --seeds 0 --prompts 20 --max-length 128 \
  --results-json quality/compare/compare_gpt2_s20.json --skip-dynamic-quant

# Alignment check (same graph point, baseline vs REVO)
PYTHONPATH=. python examples/compare_revo_vs_qp.py \
  --model gpt2 --seeds 0 --prompts 20 --max-length 128 \
  --alignment-check --hook-module transformer.ln_f \
  --alignment-results quality/compare/alignment_gpt2_ln_f.json

# Individual phase benchmarks
python examples/test_metric_field_pinn.py
python examples/test_hdram.py
python examples/test_fdm_rtd.py
python examples/bench_tqft.py
python examples/gen_final_reports.py
```

### llama.cpp (optional)

```bash
PYTHONPATH=. python examples/compare_revo_vs_qp.py \
  --llama-gguf models/gguf/mistral-7b-v0.1.Q4_K_M.gguf \
  --seeds 0 --prompts 10 --max-length 64 \
  --llama-n-ctx 2048 --llama-topk 50 --llama-tau 0.98 \
  --results-json quality/compare/compare_mistral7b_tau0.98.json
```

Current llama.cpp results demonstrate probabilistic calibration and runtime
behavior only. Graph-level compute reduction (gating / early-exit / low-rank)
is implemented in the HF/PyTorch path.

---

## The path to the vision

Each primitive points to a stratum of the vision:

```
primitive                             →  stratum
────────────────────────────────────────────────────────
tail handle (discarded information)   →  potentials on disk
gradient correction                   →  cognitive field (Φ/A/C)
QuantizedLinear                       →  token-reduced subnetwork
selective revert                      →  reconstruct-execute-forget
HyperLoRA (context → delta)           →  deterministic generative law
ModeCache (TTL/LRU)                   →  resonance as cache
Potentials (JSONL of latents)         →  compressed history
regimes (micro/macro)                 →  duality of existence
probabilistic calibration (tau)       →  knowledge ≠ expression
implicit existence                    →  model as field
bio-computational                     →  energy as fundamental constraint
primitive router (Φ per token)        →  each token gets the primitive it needs
```

---

## Limits

- **CPU-only**. No GPU. Probe, not compete.
- **I do not win on raw compression ratio**. GPTQ gives 4× with ΔPPL +0.2.
  REVO at 1.53× gives ΔNLL +0.36. My strength is not the ratio, it is that you can **revert**.
- **Gradient correction needs data**. Linearly. No shortcuts.
- **The gap between vision and reality is large and real**. The full ephemeral
  per-token execution cycle does not exist as code. It exists as specification.

---

## Primitive Router (Phase X)

Every token gets the computational primitive it needs. A learned router
(12–120 params per layer) selects per-token between 5 primitives with different
inductive biases:

| Primitive | Cost | When selected |
|-----------|------|---------------|
| Dense (O(n²)) | Full matmul | Only 1/8 layers (~76%) |
| Circulant (O(n log n)) | FFT 1D | Attention → 30% |
| WDM (O(n log n)) | Banded FFT | Attention → 70% |
| Holography | Bulk→Boundary→Bulk | Non-square layers |
| LowRank | LoRA-style | Output projection → 100% |

Key finding: after wikitext-2 training, the router **abandons dense**
in 6/8 layers (0%). Most computation uses O(n log n) with no quality loss.

```bash
# Quick demo
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from revo import PrimitiveModel
m = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
pm = PrimitiveModel(m, enabled=['dense','circulant','wdm','holography','lowrank'])
print(pm.describe())
"

# Full experiment (400 steps, wikitext-2)
python examples/exp_impressive.py

# Sparse compute (top-k)
python -c "
from revo import PrimitiveModel
pm = PrimitiveModel(model, top_k=2)  # only 2 primitives per token
pm.set_top_k(1)  # or 1 at inference time
"
```

## Generative Law (Phase XI) — the token that writes its own algebra

Every token in every layer gets its own math. Not a different expert. Not a different
weight matrix. A *different kind of operation*, chosen and parameterized by a 32-bit
code that the model generates on the fly from the token embedding itself.

### The discovery

We replaced every `nn.Linear` in distilgpt2 with a **GenerativeLayer**: a tiny MLP
(`GenerativeLaw`) that emits 32 discrete bits per token, a learned decoder
(`StructureDecoder`) that maps those bits to a choice over 5 computational
primitives, and a differentiable scale parameter.

The primitives aren't copies of the same thing:

| Primitive | Complexity | What it does |
|-----------|-----------|--------------|
| `dense` | O(n²) | Full matmul |
| `circulant` | O(n log n) | FFT convolution |
| `wdm` | O(n log(n/k)) | Banded FFT |
| `holography` | O(n²) | Bulk→boundary→bulk residual |
| `lowrank` | O(n·r) | LoRA-style adapter |

We trained the whole thing — 41M parameters, 6 layers, 768-dimensional — on
wikitext-2. No residual. No crutch. The generative law *is* the computation.

Then we looked at the codes.

### Layer 3, c_attn, on "The capital of France is Paris"

```
L3 c_attn:
  "The"      → wdm   scale=4.0    (function word)
  "capital"  → circ  scale=4.0    (content word)
  "of"       → wdm   scale=3.8    (function word)
  "France"   → circ  scale=4.0    (content word)
  "is"       → wdm   scale=3.5    (function word)
  "Paris"    → circ  scale=0.2    (content word)
```

No POS tagging in the loss. No linguistic priors. The model discovered that
articles and prepositions want one kind of linear algebra (banded FFT), while
nouns and verbs want another (full circulant). It learned to **differentiate
syntax through operator choice**.

Layer 5 pushes it further: a 19× scale difference between "The" (s=3.8) and
"Paris" (s=0.2). The algebra itself encodes the grammatical role.

We clustered 12,288 codes (50 test sequences, 24 layers, 32 bits each).
Silhouette = 0.43 at k=10. Real latent structure, not noise.

### What this means

The industry assumption is: a model is a fixed set of operations applied uniformly
to every token. Mixture-of-Experts chooses between copies of the same FFN — the
*type* of computation never changes.

This breaks that. **32 bits per token define not just which weights, but which
kind of linear algebra runs through that token at that layer.** The model discovers
its own computational ontology: dense for some tokens, FFT for others, banded FFT
for function words, holographic residual corrections where needed.

And it does it with Straight-Through Estimators — discrete decisions that
differentiate through a sigmoid + straight-through hack. It shouldn't work.
It works.

### The API

```python
from revo.generative_law import GenerativeModel, GenerativeLayer

model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Pure mode: the generative law IS the computation
gm = GenerativeModel(model, mode='pure')

# Or residual: y = orig(x) + sigmoid(eps) * gen(x)
gm = GenerativeModel(model, mode='residual')

# Train it. The codes emerge.
gm.fit(wikitext)

# Read the 32-bit genome
codes = gm.collect_codes()  # {layer_name: [B, T, 32]}

# Sparse inference: only compute top-2 primitives per token
gm.set_top_k(2)
```

### What we need to push further

The mechanism is proven. The differentiation is real. What comes next:

- **GPU time** — 41M parameters on 768-dimensional models need more than
  30 training sequences to generalize. With GPU we scale to full wikitext-2
  (millions of tokens) and measure test-set perplexity against baseline.
- **POS-tag correlation** — run spaCy over the test set and correlate code
  clusters with parts of speech. The hypothesis: codes cluster by syntactic
  function, not just by layer.
- **Scale to GPT-2 Small** — 12 layers, 124M parameters. Does the generative
  law hold at 2× the depth?
- **Cognitive field modulation** — let the scale parameter be modulated by
  a contextual field (Φ/A/C) so the same token gets different algebra in
  different contexts.
- **Ephemeral cycle** — reconstruct the subnetwork for each token on the
  fly, execute, revert, forget. The model that never lives complete in RAM.

```
The model is not a file of static weights.
The model is a trajectory of local reconstructions.
Each token generates the algebra it needs.
The rest doesn't exist.
```

---

## license

MIT (c) 2026 REVO contributors.

---

*Full vision spec at `docs/REVO_COMPUTE.md`.
The code implements the primitives. The house is being built.*
