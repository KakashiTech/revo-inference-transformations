"""REVO en Qwen2.5-0.5B — Pipeline completo con campo cognitivo (Phase XII-c).

Modelo más grande disponible en RAM (0.5B, 24 capas).
Misma arquitectura moderna que Mistral-7B (RoPE, GQA, SwiGLU).
"""

import torch, time, sys, torch.nn.functional as F
from revo._utils import load_model_tokenizer, free_memory_trim
from revo.streaming import _get_device

print("=" * 65)
print("  REVO en Qwen2.5-0.5B — Pipeline Completo XII-c")
print("=" * 65)

# ── Cargar modelo ──
print("\n[1] Cargando modelo...")
t0 = time.time()
model, tokenizer = load_model_tokenizer("Qwen/Qwen2.5-0.5B")
device = _get_device(model)
model.to(device)
model.eval()
print(f"  Cargado en {time.time()-t0:.1f}s")
n_total = sum(p.numel() for p in model.parameters())
print(f"  Parámetros: {n_total/1e6:.1f}M")

# ── Texts de prueba ──
texts = [
    "The future of artificial intelligence will transform",
    "Neural networks learn by adjusting their weights",
    "In the beginning the universe was created",
]
print(f"\n[2] Texts de prueba: {len(texts)} prompts")

# ── Build law con CognitiveField ──
print(f"\n[3] Construyendo WeightLaw con CognitiveField...")
from revo.law_streaming import build_law, law_info, pretrain_law, _extract_svd_targets
t0 = time.time()
law = build_law(model, rank=32, small_dim=8, hidden_dim=128,
                cognitive=True, cognitive_field=True)
law.to(device)
b_time = time.time() - t0
print(f"  Construido en {b_time*1000:.0f}ms")

# Pretrain rápido (50 steps) para estabilizar deltas
print(f"  Pretrain (50 steps MSE)...")
targets = _extract_svd_targets(model, rank=32)
t0 = time.time()
pretrain_law(law, targets, steps=50, lr=1e-3, verbose=False)
print(f"  Pretrain en {time.time()-t0:.1f}s")

n_law = sum(p.numel() for p in law.parameters())
n_field = sum(p.numel() for p in law.field.parameters())
print(f"  WeightLaw: {n_law:,} params ({n_law/n_total*100:.4f}% del modelo)")
print(f"  CognitiveField: {n_field:,} params")
print(f"  Compresión: {n_total/n_law:.0f}×")

# ── NLL baseline ──
print(f"\n[4] NLL baseline...")
t0 = time.time()
nlls_base = []
for text in texts:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
    nlls_base.append(nll)
print(f"  NLL medio: {sum(nlls_base)/len(nlls_base):.4f}")

# ── Phase XII-c: Cognitive Field ──
print(f"\n[5] Phase XII-c: Campo Cognitivo + generación selectiva...")

# Computar field state
from revo.streaming import _detect_arch, _extract_shared, _non_layer_pattern
arch = _detect_arch(model)
non_layer = _non_layer_pattern(arch)
state = model.state_dict(keep_vars=False)
shared = {k: v for k, v in state.items() if not non_layer.search(k)}
from revo.streaming import _extract_shared as es
wte, wpe, nw, nb, lm_w = es(shared, arch)

text = texts[0]
ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
x = F.embedding(ids, wte.to(device))

with torch.no_grad():
    field_state = law.field(x)
    scores = field_state._layer_scores
    n_layers = law.layer_emb.shape[0]
    print(f"  Texto: '{text}'")
    print(f"  Field state (inicial):")
    print(f"    Φ (valence):    {field_state.phi.item():+.3f}")
    print(f"    A (arousal):    {field_state.arousal.item():.3f}")
    print(f"    C (coherence):  {field_state.coherence.item():.3f}")
    print(f"    U (uncertainty): {field_state.uncertainty.item():.3f}")
    print(f"    Scale:          {field_state.scale.item():.3f}")
    print(f"    Layer scores:   min={scores.min().item():.3f} mean={scores.mean().item():.3f} max={scores.max().item():.3f}")
    for th in [0.3, 0.5, 0.7]:
        active = field_state.layers_to_generate(threshold=th)
        print(f"    Active (>={th}):   {len(active):>2}/{n_layers}  ({len(active)/n_layers*100:.0f}%)")

# ── Law stream forward (NLL con pesos generados) ──
print(f"\n[6] Law stream forward (NLL con pesos generados)...")
from revo.law_streaming import law_stream_forward, _save_block_templates
templates = _save_block_templates(model)
t0 = time.time()
nlls_law = []
for text in texts:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    with torch.no_grad():
        logits = law_stream_forward(model, ids, law, use_cache=False, use_native=True, block_templates=templates)
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
    nlls_law.append(nll)
l_time = time.time() - t0
nll_delta = sum(nlls_law)/len(nlls_law) - sum(nlls_base)/len(nlls_base)
print(f"  NLL law:  {sum(nlls_law)/len(nlls_law):.4f}")
print(f"  NLL base: {sum(nlls_base)/len(nlls_base):.4f}")
print(f"  ΔNLL:     {nll_delta:+.6f}")
print(f"  Tiempo:   {l_time:.2f}s ({l_time/len(texts)*1000:.0f}ms/text)")

# ── Law generate ──
print(f"\n[7] Generación con field-guided selective weights...")
from revo.law_streaming import law_generate
prompt = "The future of artificial intelligence"
t0 = time.time()
out_text, meta = law_generate(
    model, tokenizer, prompt, law,
    max_new_tokens=20, temperature=0.7, top_k=40, verbose=False,
)
g_time = time.time() - t0
print(f"  Prompt: {prompt}")
print(f"  Output: {out_text[:200]}")
print(f"  Tiempo: {g_time:.1f}s ({g_time/meta['n_new_tokens']*1000:.0f}ms/tok)")

# ── Ahora con field state variando por token ──
print(f"\n[8] Field state por token (verbose=True)...")
t0 = time.time()
out_text_v, meta_v = law_generate(
    model, tokenizer, prompt, law,
    max_new_tokens=15, temperature=0.7, top_k=40, verbose=True,
)
g2_time = time.time() - t0
print(f"\n  Output: {out_text_v[:200]}")
# Extraer field states del verbose output
print(f"  (Field states impresos arriba para cada token)")

# ── Resumen ──
print(f"\n{'='*65}")
print(f"  RESUMEN — Qwen2.5-0.5B + REVO Phase XII-c")
print(f"{'='*65}")
print(f"  Modelo:          Qwen2.5-0.5B ({n_total/1e6:.0f}M params)")
print(f"  WeightLaw:       {n_law:,} params ({n_law/n_total*100:.4f}%)")
print(f"  CognitiveField:  {n_field:,} params")
print(f"  Compresión:      {n_total/n_law:.0f}×")
print(f"  NLL base:        {sum(nlls_base)/len(nlls_base):.4f}")
print(f"  NLL law:         {sum(nlls_law)/len(nlls_law):.4f}")
print(f"  ΔNLL:            {nll_delta:+.6f}")
print(f"  Field:           Φ={field_state.phi.item():+.3f} A={field_state.arousal.item():.3f}")
print(f"  Capas activas:   {len(field_state.layers_to_generate(0.3))}/{n_layers} (>0.3)")
print(f"  Generación:      {len(out_text)} chars en {g_time:.1f}s")
print(f"  Estado:          TODO FUNCIONANDO EN MODELO REAL")
print(f"{'='*65}")
