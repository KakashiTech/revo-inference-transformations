# REVO — Research Prototype (CPU‑only)
REVO explores inference-time transformations that reshape activation dynamics without modifying model weights.

Research prototype, not a drop‑in optimizer.

This repo anchors the legitimacy of REVO as a technical line of work. It includes CPU‑only evidence, JSON artifacts, and lightweight commands to reproduce small checks. No release yet.

## What this is
- Research prototype demonstrating REVO’s final phases (VI–IX): regimes, probabilistic calibration, implicit existence, bio‑computational alignment.
- CPU‑only evaluation with reproducible artifacts under `quality/`.
- Hooks to compare baseline vs REVO at the exact same graph point (alignment‑check).

## What this is not
- Not a production library. Not a "drop‑in optimizer" replacement.
- Not a benchmark leaderboard. Numbers are small‑scale, CPU‑only probes.

## Why this matters
Most efficiency techniques trade accuracy for speed by removing information.
REVO explores the opposite direction: preserving representation geometry while changing how it is expressed over time.

## Quick evidence (CPU‑only)
Fuente: `examples/compare_revo_vs_qp.py` (gpt2, seed=0, 20 prompts). Artefactos: `quality/compare/compare_gpt2_s20.json`, `quality/compare/compare_gpt2_s20_calib.json`.

NLL vs Latency (s):

| Variant       | NLL (General) | NLL (OOD) | Eval Time (s) |
|---------------|----------------|-----------|----------------|
| baseline      | 6.1201         | 5.4191    | 9.0052         |
| quant+prune   | 6.5766         | 5.8342    | 8.9312         |
| revo (calib)  | 7.3317         | 6.5428    | 8.0398         |

Cos vs KL (recon/stability, last‑token logits):

| Variant       | recon cos | recon KL | stability cos | stability KL |
|---------------|-----------|----------|---------------|--------------|
| baseline      | 1.0000    | 0.0000   | 1.0000        | 0.0000       |
| quant+prune   | 0.99994   | 0.4067   | 0.99990       | 0.8194       |
| revo (calib)  | 0.99976   | 1.1385   | 0.99969       | 0.7052       |

Alignment‑check (mismo punto del grafo, GPT‑2): `quality/compare/alignment_summary_gpt2.json`
- Hook `transformer.ln_f`: norma L2 hidden (baseline→REVO) ≈ 231.6 → 91.4; dif L2 logits ≈ 12.46k.
- Confirma modulación/reescalado interno sin depender de cuantización.

## Reproducir (ligero, CPU‑only)
```bash
# Comparativo ligero (gpt2, 1 seed, 20 prompts)
PYTHONPATH=. python examples/compare_revo_vs_qp.py \
  --model gpt2 --seeds 0 --prompts 20 --max-length 128 \
  --results-json quality/compare/compare_gpt2_s20.json --skip-dynamic-quant

# Alignment‑check (mismo módulo en baseline vs REVO)
PYTHONPATH=. python examples/compare_revo_vs_qp.py \
  --model gpt2 --seeds 0 --prompts 20 --max-length 128 \
  --alignment-check --hook-module transformer.ln_f \
  --alignment-results quality/compare/alignment_gpt2_ln_f.json
```

---

# REVO — Compute: El modelo que vive en el tiempo

REVO Compute es una especificación abierta para ejecutar LLMs grandes en hardware mínimo sin materializar el modelo completo en RAM. La clave: el modelo no vive en la memoria; vive en el tiempo como trayectoria causal. Esta repo es 100% agnóstica de stack y libre de cualquier referencia a sistemas previos.

- Documento central: `docs/REVO_COMPUTE.md`
- Licencia: MIT
- Estado: Diseño abierto + guía de ingeniería y validación. Implementaciones de referencia son bienvenidas vía PRs.

## Qué es REVO
- Modelo como fenómeno: pesos generados efímeramente bajo leyes deterministas (Hyper‑LoRA low‑rank) y un campo cognitivo (Φ/A/C).
- Ejecución efímera: reconstruir micro‑subred por token/ventana, usar, revertir, olvidar.
- Disco como memoria lenta viva: almacenar potenciales/latentes, no pesos completos.
- Resonancia como caché: reusar modos cuando el contexto es similar.

## Uso rápido (variables estándar REVO_*)
Estas variables son genéricas y pueden mapearse a tu implementación:

- Activación núcleo:
  - `REVO_LAWS=1` activa reconstrucción efímera (Hyper‑LoRA determinista).
  - `REVO_MODECACHE=1` habilita caché de modos (`REVO_MODECACHE_TTL_SECONDS`, `REVO_MODECACHE_MAX`).
- Límites y seguridad:
  - `REVO_SCALE_MIN`, `REVO_SCALE_MAX`, `REVO_LOG_MAX_MB`.
- Campo cognitivo (Φ/A/C):
  - `REVO_FIELD_VALENCE`, `REVO_FIELD_AROUSAL`, `REVO_FIELD_COHERENCE`, `REVO_FIELD_UNCERTAINTY`.
- Modo CPU/llama.cpp (opcional):
  - `REVO_MISTRAL_BACKEND=llama_cpp`, `REVO_MISTRAL_MODEL`, `REVO_MISTRAL_CTX`, `REVO_MISTRAL_THREADS`.

Ejemplo (CPU, modelo tiny):

```bash
export REVO_LAWS=1
export REVO_MODECACHE=1
export REVO_MODECACHE_TTL_SECONDS=600
export REVO_SCALE_MIN=0.5
export REVO_SCALE_MAX=1.2
export REVO_EARLY_EXIT=1
export REVO_TOP_P=0.9
```

## Estructura
- `docs/REVO_COMPUTE.md`: Especificación completa (sanitizada y universal).
- `LICENSE`: MIT.
- `CONTRIBUTING.md`: Guía para contribuir.
- `CODE_OF_CONDUCT.md`: Código de conducta (Contributor Covenant).

## Contribuir
- Abre issues con propuestas/implementaciones.
- Envía PRs con prototipos backend‑agnóstico (Transformers, llama.cpp, etc.).
- Mantén las variables con prefijo `REVO_*` y documentación en español/inglés.

## Licencia
MIT © 2026 REVO contributors.
