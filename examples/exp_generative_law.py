#!/usr/bin/env python3
"""Experimento Phase XI: F generativa — la ley que descubre su propia álgebra.

Entrena GenerativeModel en wikitext-2 y analiza los códigos de 32 bits
que emergen. La pregunta central: ¿EL MODELO DESCUBRE UNA TAXONOMÍA
COMPUTACIONAL DEL LENGUAJE?

Hipótesis:
  - Tokens con función lingüística similar recibirán códigos similares
  - El número de clusters de códigos revela cuántas "operaciones
    fundamentales" necesita el lenguaje
  - El modelo puede aprender a usar primitivas O(n log n) sin perder calidad
"""

from __future__ import annotations
import math, time, sys
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

from revo.generative_law import GenerativeModel, StructureDecoder

DEVICE = torch.device("cpu")
MODEL = "sshleifer/tiny-gpt2"
BATCH_SIZE = 4
SEQ_LEN = 64
N_STEPS = 400
BATCHES_PER_STEP = 10
LR = 3e-4

FUNCTION_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "because", "when", "while", "though", "although",
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "as", "into",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might",
    "not", "no", "nor", "so", "very", "too", "just",
    "all", "each", "every", "both", "few", "many", "some",
    "here", "there", "where", "what", "which", "who", "whom",
    "how", "why", "than", "then", "also", "only", "still", "well",
    "up", "down", "out", "off", "over", "under", "again", "further",
}


def tokenize(examples):
    return tok(examples["text"], truncation=True, max_length=SEQ_LEN,
               padding="max_length", return_tensors="pt")


def compute_ppl(model, loader, n_batches=10):
    total_nll, total_t = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            ids = batch["input_ids"].to(DEVICE)
            out = model(ids)
            logits = out.logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   ids[:, 1:].reshape(-1), reduction="sum")
            total_nll += loss.item()
            total_t += ids[:, 1:].numel()
    return math.exp(total_nll / max(total_t, 1)), total_nll / max(total_t, 1)


def cluster_codes(codes: torch.Tensor, n_clusters: int = 10, n_iter: int = 50):
    """Simple k-means on 32-bit codes."""
    if codes.shape[0] < n_clusters:
        return None, None
    N, D = codes.shape
    # K-means++
    centroids = codes[torch.randperm(N)[:n_clusters]].clone()
    for _ in range(n_iter):
        dists = torch.cdist(codes.float(), centroids.float())  # [N, K]
        assign = dists.argmin(dim=-1)
        for k in range(n_clusters):
            mask = (assign == k)
            if mask.any():
                centroids[k] = codes[mask].float().mean(dim=0)
    return centroids, assign


# ─── Main ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("⚡ PHASE XI: F GENERATIVA — LA LEY QUE DESCUBRE SU PROPIA ÁLGEBRA ⚡")
print("=" * 70)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

# ─── Data ─────────────────────────────────────────────────────────────────

print(f"\n[0] Cargando wikitext-2...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ds = ds.filter(lambda ex: len(ex["text"].strip()) > 20)
ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
ds.set_format("torch", columns=["input_ids", "attention_mask"])
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ─── Baseline ─────────────────────────────────────────────────────────────

print(f"\n[1] Baseline (modelo original congelado)")
base_model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
base_ppl, base_nll = compute_ppl(base_model, eval_loader, n_batches=15)
print(f"    PPL: {base_ppl:.2f} | NLL: {base_nll:.4f}")

# ─── GenerativeModel ──────────────────────────────────────────────────────

print(f"\n[2] Inicializando GenerativeModel (F generativa)...")
model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
gm = GenerativeModel(
    model,
    enabled=["dense", "circulant", "wdm", "holography", "lowrank"],
    name_patterns=["c_attn", "c_proj", "c_fc"],
)
d = gm.describe()
print(f"    Capas reemplazadas: {d['replaced_layers']}")

n_law = sum(p.numel() for n, p in gm.named_parameters() if "law" in n)
n_total = sum(p.numel() for p in gm.parameters())
print(f"    Params de F (ley generativa): {n_law}")
print(f"    Params totales: {n_total:,}")
print(f"    Overhead de F: {n_law / n_total * 100:.2f}%")

for name, layer in gm._layers.items():
    print(f"      {name:<42s} {layer.in_features}→{layer.out_features}  "
          f"prims={layer._name_list}")

init_ppl, _ = compute_ppl(gm, eval_loader, n_batches=15)
print(f"\n    PPL inicial (F aleatoria): {init_ppl:.2f}")

# ─── Train ────────────────────────────────────────────────────────────────

print(f"\n[3] Entrenando TODOS los parámetros ({N_STEPS} steps)...")
opt = torch.optim.AdamW(gm.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

ppl_log = [(0, init_ppl, base_ppl)]
t_start = time.time()

for step in range(1, N_STEPS + 1):
    gm.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        if i >= BATCHES_PER_STEP:
            break
        ids = batch["input_ids"].to(DEVICE)
        opt.zero_grad()
        out = gm(ids)
        logits = out.logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    scheduler.step()

    if step % 100 == 0 or step == N_STEPS:
        cur_ppl, cur_nll = compute_ppl(gm, eval_loader, n_batches=15)
        ppl_log.append((step, cur_ppl, base_ppl))
        elapsed = time.time() - t_start
        print(f"    step {step:4d}/{N_STEPS} | loss={total_loss/BATCHES_PER_STEP:.4f} | "
              f"PPL={cur_ppl:.2f} (Δ={cur_ppl-base_ppl:+.2f}) | {step/elapsed:.1f} step/s")

final_ppl, _ = compute_ppl(gm, eval_loader, n_batches=15)
print(f"\n[4] Resultados PPL")
print(f"    Baseline frozen: {base_ppl:>10.2f}")
print(f"    F aleatoria:     {init_ppl:>10.2f}  (Δ={init_ppl-base_ppl:+.2f})")
print(f"    F entrenada:     {final_ppl:>10.2f}  (Δ={final_ppl-base_ppl:+.2f})")
print(f"    Mejora:          {init_ppl-final_ppl:>+10.2f} nats")

# ─── Code Analysis ───────────────────────────────────────────────────────

print(f"\n[5] ANÁLISIS DE CÓDIGOS — ¿qué estructura computacional descubrió F?")
gm.eval()

# Collect codes over many tokens
all_codes: Dict[str, torch.Tensor] = defaultdict(list)
all_tokens: Dict[str, List[str]] = defaultdict(list)
all_func: Dict[str, List[bool]] = defaultdict(list)

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 30:
            break
        ids = batch["input_ids"].to(DEVICE)
        _ = gm(ids)
        codes = gm.collect_codes()
        # Decode tokens
        for name, c in codes.items():
            all_codes[name].append(c)  # c: [B, T, 32]
            for bi in range(c.shape[0]):
                for ti in range(c.shape[1]):
                    t_id = ids[bi, ti].item()
                    t_str = tok.decode(t_id).strip().lower()
                    all_tokens[name].append(t_str)
                    all_func[name].append(t_str in FUNCTION_WORDS)

# Concatenate and analyze per layer
print(f"\n{'Layer':<42s} {'N':>6s} {'Clusters':>9s} {'%FuncPrim1':>11s} {'%FuncPrim2':>11s} {'%ContPrim1':>11s} {'%ContPrim2':>11s}")
print("-" * 100)

for name in sorted(all_codes.keys()):
    codes_cat = torch.cat(all_codes[name], dim=0)  # [N, 32]
    N = codes_cat.shape[0]

    if N < 5:
        continue

    # K-means clustering of 32-bit codes
    n_clusters = min(8, N // 2)
    centroids, assign = cluster_codes(codes_cat, n_clusters=n_clusters)

    token_list = all_tokens[name]
    func_list = all_func[name]

    # For each cluster, show top tokens and function word ratio
    cluster_info = []
    if assign is not None:
        for ci in range(n_clusters):
            mask = (assign == ci)
            n_in = mask.sum().item()
            if n_in == 0:
                continue
            func_ratio = sum(f for f, m in zip(func_list, mask.tolist()) if m) / n_in
            # Top tokens in this cluster
            toks_in = [t for t, m in zip(token_list, mask.tolist()) if m]
            top_toks = sorted(set(toks_in), key=toks_in.count, reverse=True)[:3]
            cluster_info.append((ci, n_in, func_ratio, top_toks))

        n_clusters_found = len(cluster_info)
    else:
        n_clusters_found = 0

    # Decode average codes to see primitives
    code_mean = codes_cat.float().mean(dim=0)
    decoded_mean = StructureDecoder.decode(code_mean.unsqueeze(0), 5)
    p1_mean = decoded_mean["prim_idx_1"][0].item()
    p2_mean = decoded_mean["prim_idx_2"][0].item()

    # Function vs content primitive preference
    func_codes = codes_cat[torch.tensor(func_list, dtype=torch.bool)]
    cont_codes = codes_cat[~torch.tensor(func_list, dtype=torch.bool)]
    if len(func_codes) > 3 and len(cont_codes) > 3:
        f_dec = StructureDecoder.decode(func_codes.mean(dim=0, keepdim=True), 5)
        c_dec = StructureDecoder.decode(cont_codes.mean(dim=0, keepdim=True), 5)
        fp1 = f_dec["prim_idx_1"][0].item()
        fp2 = f_dec["prim_idx_2"][0].item()
        cp1 = c_dec["prim_idx_1"][0].item()
        cp2 = c_dec["prim_idx_2"][0].item()
    else:
        fp1 = fp2 = cp1 = cp2 = -1

    print(f"{name:<42s} {N:>6d} {n_clusters_found:>4d}/{n_clusters:>3d} "
          f"{fp1:>6.1f} ({fp2:>4.1f})    {cp1:>6.1f} ({cp2:>4.1f})")

    # Show cluster details for first few layers
    if list(all_codes.keys()).index(name) < 3:
        for ci, n_in, func_ratio, top_toks in cluster_info[:4]:
            toks_str = ", ".join(top_toks)
            func_pct = func_ratio * 100
            print(f"      cluster {ci}: {n_in:4d} tokens  "
                  f"func={func_pct:.0f}%  top: [{toks_str}]")

# ─── Token-level tour ────────────────────────────────────────────────────

print(f"\n[6] TOUR POR TOKEN — códigos individuales")
test_sents = [
    "The theory of relativity changed physics forever",
    "She walked to the store and bought fresh milk",
    "In the beginning God created the heavens and the earth",
]

gm.eval()
for sent in test_sents:
    print(f"\n  > {sent}")
    enc = tok(sent, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = gm(**enc)
    codes = gm.collect_codes()

    # Show first layer's codes
    first_name = list(codes.keys())[0]
    first_codes = codes[first_name][0]  # [T, 32]
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])

    for ti in range(min(6, len(tokens))):
        c = first_codes[ti]
        decoded = StructureDecoder.decode(c.unsqueeze(0), 5)
        p1 = int(decoded["prim_idx_1"][0].item())
        p2 = int(decoded["prim_idx_2"][0].item())
        w = decoded["weight"][0].item()
        prim_names = ["dense", "circ", "wdm", "holo", "lora"]
        p1n = prim_names[p1] if p1 < len(prim_names) else "?"
        p2n = prim_names[p2] if p2 < len(prim_names) else "?"
        tag = "FUNC" if tokens[ti].strip().lower() in FUNCTION_WORDS else "CONT"
        print(f"    tok {ti:2d} '{tokens[ti]:>12s}' [{tag}] → "
              f"prim1={p1n} prim2={p2n}  w={w:.2f}  "
              f"code={c[:8].int().tolist()}...")

# ─── Efficiency ───────────────────────────────────────────────────────────

print(f"\n[7] EFICIENCIA — primitivas seleccionadas por capa")
with torch.no_grad():
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        _ = gm(ids)
        break
    codes = gm.collect_codes()

prim_names = ["dense", "circ", "wdm", "holo", "lora"]
for name, c in codes.items():
    decoded = StructureDecoder.decode(c.reshape(-1, 32), 5)
    p1_vals = decoded["prim_idx_1"].float()  # [N]
    p2_vals = decoded["prim_idx_2"].float()
    # Round to nearest integer for counting
    p1_int = p1_vals.round().long().clamp(0, 4)
    p2_int = p2_vals.round().long().clamp(0, 4)
    p1_counts = torch.bincount(p1_int, minlength=5).float()
    p2_counts = torch.bincount(p2_int, minlength=5).float()
    total_counts = p1_counts + p2_counts
    total = total_counts.sum()
    if total > 0:
        pcts = " ".join(f"{pn}={total_counts[i].item()/total*100:.0f}%"
                        for i, pn in enumerate(prim_names))
        print(f"  {name}: {pcts}")

# ─── Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RESUMEN: F GENERATIVA")
print("=" * 70)
print(f"  • {d['replaced_layers']} capas con F generativa (MLP → 32-bit code)")
print(f"  • {n_law} params de F controlan {n_total:,} params totales")
print(f"  • PPL: {base_ppl:.0f} (baseline) → {init_ppl:.0f} (inicial) → {final_ppl:.0f} (entrenado)")
print(f"  • Códigos de 32 bits: {2**32:,} programas posibles por token")
print(f"  • Tiempo total: {time.time()-t_start:.0f}s")
print("=" * 70)
print("⚡ F GENERATIVA: EL MODELO QUE INVENTA SU PROPIA ÁLGEBRA ⚡")
