# REVO — el modelo que vive en el tiempo

**REVO** (Reversible Execution Via reOrdering) no es una técnica de compresión.
Es un cambio de definición: el modelo no es un archivo de pesos estáticos.
Es una **ley de transformación reproducible** que existe en el tiempo como trayectoria causal.

La inferencia no es ejecución global de una red fija.
Es **reconstrucción local bajo contexto**: la subred que necesitas para este token,
generada efímeramente, ejecutada, revertida, olvidada.

El modelo como fenómeno, no como cosa.

```
hoy:     deltas reversibles + corrección por gradiente + cuantización lossless
           ↓
mañana:  reconstrucción efímera por token, campo cognitivo modulando la trayectoria,
         el disco como memoria lenta viva, el modelo que no pesa nada en RAM
```

**Status:** research prototype — 9 phases implemented, CPU-only, 43 tests (pytest),
CI via GitHub Actions. All modules verified importable.

This repo anchors the legitimacy of REVO as a technical line of work. It includes
CPU-only evidence, JSON artifacts, and lightweight commands to reproduce small
checks. No release yet.

---

## la ruptura

El axioma de la industria es: el modelo es un objeto que debe vivir en RAM.
De ahí viene todo — cuantización, poda, destilación. Sacrificar fidelidad para que
quepa. Y aceptar que ese sacrificio es permanente.

Eso es falso. Un modelo grande no necesita existir completo en ningún instante.
Basta con que sea una trayectoria causal reproducible. Como el océano: no tienes
todas las ondas en un punto, pero el océano se comporta coherentemente.

REVO es ese cambio de paradigma.

---

## el modelo como fenómeno (cuatro estratos)

Un modelo REVO no es una pila de matrices. Es cuatro capas que nunca coexisten completas:

```
1. LEY GENERATIVA         CPU, RAM fija (MB)
   Funciones compactas que generan deltas de pesos desde contexto latente.
   Hyper-LoRA determinista: Z → (A, B, scale).

2. CAMPO COGNITIVO Φ/A/C  RAM viva (KB-MB)
   Valencia, arousal, coherencia. Decide qué reconstruir, con qué magnitud,
   a qué velocidad. El campo modula la trayectoria en tiempo de inferencia.

3. HISTORIA COMPRIMIDA    Disco (GB)
   Potenciales, latentes, mapas de resonancia, trayectorias.
   No weights.bin — mapas latentes, campos comprimidos, feromonas semánticas.

4. MANIFESTACIÓN LOCAL     RAM efímera (KB)
   Micro-subred reconstruida para este token/ventana. Se ejecuta, se revierte,
   se destruye. Nunca existe más que un fragmento ínfimo del modelo a la vez.
```

El truco: **nunca existen 100% de los parámetros a la vez**.
Lo que existe es la certeza de que, dado el contexto correcto,
la reconstrucción será coherente.

---

## la primitiva fundamental: el delta

REVO no reemplaza pesos. REVO aplica **deltas** — perturbaciones low-rank
reversibles. Cada delta captura una transformación, y puede ser revertido
con pérdida estadísticamente indetectable (~3e-06).

De esta primitiva nace todo:

```
delta              →  cualquier transformación es reversible
tail handle        →  la información descartada no se destruye, se guarda
gradient correction →  la calidad perdida se recupera con backprop
selección modular  →  solo reviertes lo que duele
```

El ciclo completo:

1. Comprimes (SVD truncation, cuantización, poda espectral)
2. El handle guarda lo que descartaste (factores low-rank, pesos originales)
3. Medís impacto en NLL
4. Si duele → gradient correction o revert selectivo
5. Si no duele → listo, el modelo vive comprimido pero eres libre de volver

Ninguna otra técnica puede hacer esto.
GPTQ no puede. SparseGPT no puede. AWQ no puede.
Todas destruyen permanentemente. REVO no.

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

GPT-2 124M, wikitext-2, truncamiento al 25% del rango singular.

```
Config                                    ΔNLL      Compresión  Revertible
───────────────────────────────────────────────────────────────────────────
SVD puro                                 +4.09      1.53×       ✅ 3e-06
+ gradient correction (5 pasos AdamW)    +0.36      1.53×       ✅
+ calibración separada (100 txts)        +2.01      1.53×       ✅
safe_compress (revert selectivo)         +0.65      variable    ✅
```

La corrección por gradiente cierra ~90% de la brecha de calidad.

### QuantizedLinear 4-bit

| Config               | ΔNLL      | Compresión  | Revertible |
|----------------------|-----------|-------------|------------|
| 4-bit group=32       | −0.005*   | 8.0×        | ✅ 1e-08   |
| 4-bit group=64       | +0.040    | 8.0×        | ✅         |
| 3-bit group=32       | +0.345    | 10.7×       | ✅         |
| 2-bit group=64       | +4.15     | 16.0×       | ✅         |

*\* Mejora por regularización. 4-bit lossless.*

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
│
├── pipeline.py           # Orchestrator: runs all phases, collects metrics
├── unified_main.py       # Unified CLI entry point for all phases
├── archive/              # Archived experimental modules (15+)
│
├── _logging.py           # Centralised logging configuration
└── _utils.py             # Shared utilities (seed, NLL, module iteration)

examples/                 # Runnable benchmarks and tests (15+ scripts)
tests/                    # 43 pytest tests across 6 suites
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

---

## Reproducir (CPU-only)

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

## el camino hacia la visión

Cada primitiva apunta a un estrato de la visión:

```
primitiva                             →  estrato
────────────────────────────────────────────────────────
tail handle (información descartada)  →  potenciales en disco
gradient correction                   →  campo cognitivo (Φ/A/C)
QuantizedLinear                       →  subred reducida por token
selective revert                      →  reconstruir-ejecutar-olvidar
HyperLoRA (contexto → delta)          →  ley generativa determinista
ModeCache (TTL/LRU)                   →  resonancia como caché
Potentials (JSONL de latentes)        →  historia comprimida
regímenes (micro/macro)               →  dualidad de existencia
calibración probabilística (tau)      →  conocimiento ≠ expresión
existencia implícita                  →  modelo como campo
biocomputacional                      →  energía como restricción fundamental
```

---

## límites

- **CPU-only**. Sin GPU. Sondear, no competir.
- **No gano en ratio de compresión puro**. GPTQ da 4× con ΔPPL +0.2.
  REVO a 1.53× da ΔNLL +0.36. Mi fortaleza no es el ratio, es que puedes **volver**.
- **Gradient correction necesita datos**. Linealmente. Sin atajos.
- **La brecha visión-realidad es grande y real**. El ciclo completo de ejecución
  efímera por token no existe como código. Existe como especificación.

---

## próximos pasos

1. **Error diffusion en QuantizedLinear** — mejora sobre min-max ya demostrada.
2. **Campo cognitivo (Φ/A/C) experimental** — modulación de scales por contexto.
3. **Ciclo efímero completo** — reconstruir subred por token, ejecutar, revertir.
4. **Disco como memoria lenta** — handles REVO en disco, no en RAM.

---

## licencia

MIT (c) 2026 REVO contributors.

---

*Especificación completa de la visión en `docs/REVO_COMPUTE.md`.
El código implementa las primitivas. La casa se está construyendo.*
