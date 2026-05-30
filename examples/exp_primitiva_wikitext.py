#!/usr/bin/env python3
"""Entrena el router en wikitext-2 y mide perplejidad vs baseline."""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from revo.primitiva_router import PrimitiveModel

DEVICE = torch.device("cpu")
MODEL = "sshleifer/tiny-gpt2"
BATCH_SIZE = 4
SEQ_LEN = 64
N_STEPS = 500
LR = 3e-3

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

def tokenize(examples):
    return tok(examples["text"], truncation=True, max_length=SEQ_LEN,
               padding="max_length", return_tensors="pt")

def ppl(model, loader, n_batches=10):
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

# Load wikitext
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                  trust_remote_code=True)
ds = ds.filter(lambda ex: len(ex["text"].strip()) > 20)
ds.set_format("torch", columns=["input_ids", "attention_mask"])
loader = torch.utils.data.DataLoader(
    ds.map(tokenize, batched=True, remove_columns=ds.column_names),
    batch_size=BATCH_SIZE, shuffle=True,
)

print("=== Baseline (modelo original) ===")
base = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
base_ppl, _ = ppl(base, loader)
print(f"  PPL: {base_ppl:.2f}")

print("\n=== PrimitiveModel + entrenar router ===")
model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
pm = PrimitiveModel(
    model,
    enabled=["dense", "circulant", "wdm", "holography", "lowrank"],
    name_patterns=["c_attn", "c_proj", "c_fc"],
    router_hard=True,
)
print(f"  Capas: {pm.describe()['replaced_layers']}")

init_ppl, _ = ppl(pm, loader)
print(f"  PPL inicial: {init_ppl:.2f}")

opt = torch.optim.AdamW(
    [p for n, p in pm.named_parameters() if "router" in n],
    lr=LR,
)

for step in range(1, N_STEPS + 1):
    pm.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        opt.zero_grad()
        out = pm(ids)
        logits = out.logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= 20:  # 20 batches per step
            break
    if step % 100 == 0 or step == N_STEPS:
        cur_ppl, _ = ppl(pm, loader)
        print(f"  step {step:4d} | loss={total_loss/n_batches:.4f} | PPL={cur_ppl:.2f} (Δ={cur_ppl-base_ppl:+.2f})")

final_ppl, _ = ppl(pm, loader)
print(f"\n=== Resumen ===")
print(f"  Baseline:   {base_ppl:.2f}")
print(f"  Inicial:    {init_ppl:.2f}")
print(f"  Final:      {final_ppl:.2f}")
print(f"  Mejora:     {init_ppl - final_ppl:+.2f} nats")
print(f"  vs baseline: {final_ppl - base_ppl:+.2f}")

# Análisis final del router
with torch.no_grad():
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        _ = pm(ids)
        break
    w = pm.collect_router_weights()
prim_names = ["dense", "circ", "wdm", "holo", "lora"]
for name, wt in w.items():
    probs = wt.mean(dim=tuple(range(wt.ndim - 1)))
    parts = " ".join(f"{pn}={probs[i].item()*100:.0f}%" for i, pn in enumerate(prim_names) if i < len(probs))
    print(f"  {name}: {parts}")
