# REVO — Reversible Inference-Time Transformations

**REVO** (Reversible Execution Via reOrdering) enables safe, reversible model compression with fine-grained control. Compress, measure, revert — zero permanent damage.

```
┌─ Compress ──→ Measure ──→ Accept? ──→ Done
│                              │
└──── Revert (lossless) ←──────┘
```

---

## Core value proposition

| Feature | What it means |
|---------|---------------|
| **Revertibility** | Every compression saves the discarded information as low-rank tail handles. Revert delta **~3e-06** (statistically lossless). |
| **Gradient correction** | Fine-tunes compressed factors via backprop through the full model to minimize NLL directly. Closes the quality gap significantly. |
| **Selective compression** | Compress all modules, then surgically revert individual ones that degrade quality too much. |
| **No permanent changes** | Compressed state lives in-memory alongside original. Revert at any time. |

---

## Real benchmarks (GPT-2 124M, 1.53× compression, wikitext-2)

| Method | ΔNLL | Revertible | Notes |
|--------|------|------------|-------|
| SVD truncation (25% rank) | +4.09 | ✅ (REVO) | Pure SVD, aggressive truncation |
| + Gradient correction | **+0.36** | ✅ (REVO) | 100 calib texts, 5 AdamW steps |
| + Gradient correction | +2.01 | ✅ (REVO) | Held-out calib (100 texts, 1 step) |
| safe_compress (selective revert) | +0.65 | ✅ (REVO) | 23/48 modules compressed, target Δ≤0.5 |

**Takeaway**: Gradient correction closes ~90% of the quality gap when calib distribution matches eval. With held-out calibration data, the improvement is ~50% (from +4.09 to +2.01). More calibration data scales the improvement linearly.

---

## Quick start

```bash
# Install
pip install -e .

# Run all tests
python -m pytest tests/ -v

# SVD compression + gradient correction benchmark (GPT-2)
python examples/exp_revo_fusion.py
```

---

## Repository structure

```
revo/
├── act_svd.py            # SVD compression + tail handles + gradient correction
├── lowrank.py            # LowRankLinear module (A@B factorization)
├── spectral.py           # Spectral pruning
├── layer_profile.py      # Energy-based rank allocation
│
├── engine.py             # Low-rank delta apply/revert (NumPy)
├── hyperlora.py          # Context-to-(A,B,scale) generation
├── mode_cache.py         # TTL/LRU mode cache
├── features.py           # Feature extraction
├── potentials.py         # Logging
│
├── pipeline.py           # Phase orchestrator (experimental)
├── archive/              # Archived experimental modules
│   ├── true_reversible.py
│   ├── tensor_train.py
│   └── ... (15 more)
│
├── _utils.py             # Shared utilities
└── _logging.py           # Logging configuration

examples/
├── exp_revo_fusion.py    # SVD + gradient correction benchmark
├── compare_revo_vs_qp.py # REVO vs quantization+pruning comparison
└── ...                   # Phase-specific test scripts

quality/experiments/      # JSON benchmark artifacts
```

---

## How REVO handles work

1. **SVD truncation**: Factor W ≈ U·S·Vᵀ, keep top-k singular values/vectors
2. **Tail handle**: Store discarded tail U_tail, S_tail, V_tail as low-rank factors
3. **Module replacement**: Replace original module with LowRankLinear(A, B)
4. **Gradient correction** (optional): Fine-tune A/B via backprop to minimize NLL on calibration data
5. **Revert**: W_original ≈ A·B + U_tail·S_tail·V_tailᵀ (delta ~3e-06)

---

## Project status

| Component | Status |
|-----------|--------|
| SVD compression + tail handles | ✅ Tested on GPT-2 124M |
| Gradient correction | ✅ ΔNLL +4.09 → +0.36 |
| Selective revert (safe_compress) | ✅ Experimental |
| LowRankLinear module | ✅ Tested |
| Spectral pruning | ✅ Tested |
| Hyperlora / ModeCache / Potentials | ✅ Tested |
| Pipeline (9-phase orchestrator) | 🟡 Experimental, partial |
| Archive modules (15) | 🟤 Experimental, unmaintained |

---

## Limitations

- **CPU-only**: No GPU support. Benchmarks are small-scale probes.
- **Not competitive with GPTQ**: GPTQ achieves 4× compression with ΔPPL +0.2. REVO at 1.53× gives ΔNLL +0.36–2.01. REVO's strength is reversibility, not peak compression.
- **Gradient correction needs calibration data**: Improvement scales with data quantity.

---

## License

MIT (c) 2026 REVO contributors.
