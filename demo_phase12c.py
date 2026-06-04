"""REVO Phase XII-c: Cognitive Field + Selective Weight Generation.

The model generates its own weights differently per token, governed by
a Φ/A/C cognitive field that decides which layers need regeneration.

Three-tier architecture:
  Φ/A/C Field (8 params) → what to reconstruct + magnitude
  WeightLaw (86K params)  → how to reconstruct (low-rank deltas)
  ModeCache (64 slots)    → what to reuse (resonance as cache)

Run:
    python demo_phase12c.py            # SmolLM2-135M
    python demo_phase12c.py tiny       # tiny-gpt2 quick test
"""

import sys, time, torch, torch.nn.functional as F
from revo.law_streaming import (
    build_law, law_generate, law_stream_forward,
    CognitiveField, FieldState, WeightModeCache,
)
from revo._utils import load_model_tokenizer, measure_memory_rss
from revo.streaming import _get_device, _detect_arch, _extract_shared, _non_layer_pattern


def demo(model_name: str, rank: int = 4, small_dim: int = 2,
         hidden_dim: int = 64, tokens: int = 30, tiny: bool = False):

    print(f"Loading {model_name}...")
    model, tokenizer = load_model_tokenizer(model_name)
    device = _get_device(model)

    # ─── Build law with cognitive field ──────────────────────────────
    law = build_law(model, rank=rank, small_dim=small_dim,
                    hidden_dim=hidden_dim, cognitive=True,
                    cognitive_field=True).to(device)
    n_law = sum(p.numel() for p in law.parameters())
    n_field = sum(p.numel() for p in law.field.parameters()) if hasattr(law, 'field') else 0
    print(f"\n{'='*60}")
    print(f"  WeightLaw:    {n_law:,} params")
    print(f"  CognitiveField: {n_field:,} params")
    print(f"  Delta mode:   {law.rank <= 8} (rank={law.rank})")
    print(f"{'='*60}")

    # ─── Show field state per token ──────────────────────────────────
    enc = tokenizer("The future of artificial intelligence",
                    return_tensors="pt").input_ids.to(device)
    state_dict = model.state_dict(keep_vars=False)
    arch = _detect_arch(model)
    non_layer = _non_layer_pattern(arch)
    shared = {k: v for k, v in state_dict.items() if not non_layer.search(k)}
    wte, wpe, nw, nb, lm_w = _extract_shared(shared, arch)
    x = F.embedding(enc, wte.to(device))

    field_state = law.field(x)
    scores = field_state._layer_scores
    active_03 = field_state.layers_to_generate(threshold=0.3)
    active_05 = field_state.layers_to_generate(threshold=0.5)
    active_07 = field_state.layers_to_generate(threshold=0.7)

    n_layers_model = len(scores)
    print(f"\n  Cognitive Field State (initial prompt):")
    print(f"    Φ (valence):    {field_state.phi.item():+.3f}  [-1, +1]")
    print(f"    A (arousal):    {field_state.arousal.item():.3f}  [0, 1]")
    print(f"    C (coherence):  {field_state.coherence.item():.3f}  [0, 1]")
    print(f"    U (uncertainty): {field_state.uncertainty.item():.3f}  [0, 1]")
    print(f"    Scale:          {field_state.scale.item():.3f}")
    print(f"    Layer scores:   min={scores.min().item():.2f} mean={scores.mean().item():.2f} max={scores.max().item():.2f}")
    print(f"    Active layers:")
    print(f"      threshold=0.3: {len(active_03)}/{n_layers_model} layers")
    print(f"      threshold=0.5: {len(active_05)}/{n_layers_model} layers")
    print(f"      threshold=0.7: {len(active_07)}/{n_layers_model} layers")

    # ─── Quality check ───────────────────────────────────────────────
    with torch.no_grad():
        logits_full = model(enc).logits
    logits_law = law_stream_forward(model, enc, law, use_cache=False, use_native=True)
    nll_full = F.cross_entropy(logits_full[:, :-1].reshape(-1, logits_full.shape[-1]),
                                enc[:, 1:].reshape(-1)).item()
    nll_law = F.cross_entropy(logits_law[:, :-1].reshape(-1, logits_law.shape[-1]),
                               enc[:, 1:].reshape(-1)).item()
    print(f"\n  Quality:")
    print(f"    Full NLL: {nll_full:.4f}")
    print(f"    Law NLL:  {nll_law:.4f}  (Δ={nll_law-nll_full:+.4f})")

    # ─── Generate with cognitive field ───────────────────────────────
    prompt = "The future of artificial intelligence"
    print(f"\n  Generating {tokens} tokens with field-guided selective weights...")
    t0 = time.time()
    out_text, meta = law_generate(model, tokenizer, prompt, law,
                                   max_new_tokens=tokens, temperature=0.7,
                                   top_k=40, verbose=False)
    t_gen = time.time() - t0
    print(f"    {t_gen/tokens*1000:.0f}ms/tok  ({t_gen:.1f}s total)")
    print(f"\n  Field-governed generation:")
    print(f"    Prompt: {prompt}")
    print(f"    Output: {out_text[:250]}")

    # ─── Field variation across tokens ───────────────────────────────
    print(f"\n  Field state varied across {tokens} tokens:")
    print(f"    (Check verbose=True in law_generate for per-token field print)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if mode == "tiny":
        demo("sshleifer/tiny-gpt2", rank=4, small_dim=2,
             hidden_dim=64, tokens=15, tiny=True)
    else:
        demo("HuggingFaceTB/SmolLM2-135M", rank=4, small_dim=2,
             hidden_dim=64, tokens=30)
