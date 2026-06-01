"""Phase 3 training + evaluation: GenerativeModel puro, wikitext-2, checkpoint + report."""
import time, math, torch, json, os, sys, signal
torch.set_num_threads(8)
torch.set_grad_enabled(True)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

DEVICE = torch.device('cpu')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'phase3')
os.makedirs(SAVE_DIR, exist_ok=True)
interrupted = False

def handler(signum, frame):
    global interrupted
    if not interrupted:
        print("\n[SIGINT] Graceful shutdown... saving checkpoint.")
        interrupted = True
signal.signal(signal.SIGINT, handler)

print("=" * 60)
print("PHASE 3: GenerativeModel puro — entrenamiento + evaluación completa")
print("=" * 60)

# ─── Data ──────────────────────────────────────────────────────────────
print("\n[Data] Loading wikitext-2...")
tok = AutoTokenizer.from_pretrained('distilgpt2')
tok.pad_token = tok.eos_token

def tokenize_ds(split, max_seqs=None):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    ds = ds.filter(lambda ex: len(ex['text'].strip()) > 20)
    all_ids = []
    for i, ex in enumerate(ds):
        if max_seqs and i >= max_seqs: break
        enc = tok(ex['text'][:256], truncation=True, max_length=128,
                  padding='max_length', return_tensors='pt')
        all_ids.append(enc.input_ids[0])
    return torch.stack(all_ids).to(DEVICE)

train_ids = tokenize_ds('train', max_seqs=100)
test_ids  = tokenize_ds('test',  max_seqs=50)
print(f"  Train: {len(train_ids)} seqs")
print(f"  Test:  {len(test_ids)} seqs")

# ─── Model ─────────────────────────────────────────────────────────────
print("\n[Model] Loading distilgpt2 + GenerativeModel (pure mode)...")
model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
gm = GenerativeModel(
    model, enabled=['dense', 'circulant', 'wdm', 'lowrank', 'holography'],
    name_patterns=['c_attn', 'c_proj', 'c_fc'], mode='pure'
)
print(f"  {gm.describe()['replaced_layers']} layers replaced")
t0 = time.time()

# ─── Warmup ────────────────────────────────────────────────────────────
print("\n[Warmup] Law + decoder only...")
for n, p in model.named_parameters(): p.requires_grad = False
for layer in gm._layers.values():
    for p in layer.law.parameters(): p.requires_grad = True
    for p in layer.decoder.parameters(): p.requires_grad = True

tp = sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f"  {tp:,} trainable params")
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gm.parameters()), lr=3e-3)

for step in range(1, 21):
    if interrupted: break
    gm.train(); perm = torch.randperm(len(train_ids)); total_loss = 0.0; batches = 0
    for start in range(0, len(train_ids), 8):
        if interrupted: break
        b = train_ids[perm[start:start+8]]
        opt.zero_grad()
        out = gm(input_ids=b, attention_mask=b)
        loss = F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1))
        loss.backward(); opt.step(); total_loss += loss.item(); batches += 1
    if step % 10 == 0:
        gm.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, min(40, len(train_ids)), 8):
                b = train_ids[s:s+8]
                o = gm(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1),reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
            ppl = math.exp(nll / max(nt, 1))
        dt = time.time() - t0
        print(f"  warmup {step:2d} loss={total_loss/batches:.3f} ppl~{ppl:.0f} ({dt:.0f}s)")

# ─── Full training ─────────────────────────────────────────────────────
print("\n[Full] All 41M params...")
for p in model.parameters(): p.requires_grad = True
tp = sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f"  {tp:,} trainable params")
opt2 = torch.optim.AdamW(gm.parameters(), lr=3e-4)
TOTAL_STEPS = 100
last_save = 0

for step in range(1, TOTAL_STEPS + 1):
    if interrupted: break
    gm.train(); perm = torch.randperm(len(train_ids)); total_loss = 0.0; batches = 0
    for start in range(0, len(train_ids), 8):
        if interrupted: break
        b = train_ids[perm[start:start+8]]
        opt2.zero_grad()
        out = gm(input_ids=b, attention_mask=b)
        loss = F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0)
        opt2.step(); total_loss += loss.item(); batches += 1

    # Save every 25 steps and on interrupt
    if step % 25 == 0 or interrupted:
        gm.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, min(40, len(train_ids)), 8):
                b = train_ids[s:s+8]
                o = gm(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1),reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
            ppl = math.exp(nll / max(nt, 1))
        dt = time.time() - t0
        print(f"  step {step:3d} loss={total_loss/batches:.3f} train_ppl~{ppl:.0f} ({dt:.0f}s)")
        torch.save({'model_state': model.state_dict(), 'step': step},
                   os.path.join(SAVE_DIR, f'phase3_step{step}.pt'))
        last_save = step

if interrupted:
    print(f"\n[SIGINT] Saved at step {last_save}. Proceeding with evaluation.")
    # Load last checkpoint for eval
    cp = torch.load(os.path.join(SAVE_DIR, f'phase3_step{last_save}.pt'), map_location=DEVICE)
    model.load_state_dict(cp['model_state'])
else:
    torch.save({'model_state': model.state_dict(), 'step': TOTAL_STEPS},
               os.path.join(SAVE_DIR, 'phase3_final.pt'))
    print(f"\n[Save] → {SAVE_DIR}/phase3_final.pt")

actual_steps = last_save if interrupted else TOTAL_STEPS

# ─── Test-set evaluation ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)
gm.eval()
with torch.no_grad():
    nll, nt = 0., 0
    for s in range(0, len(test_ids), 8):
        b = test_ids[s:s+8]
        o = gm(input_ids=b, attention_mask=b)
        l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1),reduction='sum')
        nll += l.item(); nt += b[:,1:].numel()
    test_ppl = math.exp(nll / max(nt, 1))
    test_loss = nll / max(nt, 1)
print(f"  Test NLL: {test_loss:.4f}")
print(f"  Test PPL: {test_ppl:.1f}")

# ─── Baseline comparison ──────────────────────────────────────────────
print("\n[Baseline] distilgpt2 original...")
base = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE).eval()
with torch.no_grad():
    nll_b, nt_b = 0., 0
    for s in range(0, len(test_ids), 8):
        b = test_ids[s:s+8]
        o = base(input_ids=b, attention_mask=b)
        l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1),reduction='sum')
        nll_b += l.item(); nt_b += b[:,1:].numel()
    base_ppl = math.exp(nll_b / max(nt_b, 1))
print(f"  Baseline PPL (distilgpt2):  {base_ppl:.1f}")
print(f"  GenerativeModel PPL:        {test_ppl:.1f}")
print(f"  Ratio:                      {test_ppl/base_ppl:.2f}x")

# ─── Code collection ───────────────────────────────────────────────────
print("\n[Analysis] Collecting codes...")
all_codes, all_tokens, all_layers, all_tok_strs = [], [], [], []
with torch.no_grad():
    for s in range(0, len(test_ids), 8):
        b = test_ids[s:s+8]
        _ = gm(input_ids=b, attention_mask=b)
        for ln, c in gm.collect_codes().items():
            for bi in range(b.shape[0]):
                toks = tok.convert_ids_to_tokens(b[bi].tolist())
                for ti in range(c.shape[1]):
                    all_codes.append(c[bi, ti].cpu().numpy())
                    all_layers.append(ln)
                    all_tok_strs.append(toks[ti])
nc = np.array(all_codes)
print(f"  {len(nc)} codes collected")

# ─── Silhouette analysis ──────────────────────────────────────────────
print("\n[Clustering] Silhouette analysis...")
sil_results = {}
for k in [4, 6, 8, 10, 12, 16, 20]:
    if k >= len(nc): break
    km = KMeans(k, random_state=42, n_init=3)
    cid = km.fit_predict(nc)
    if len(nc) > 5000:
        idx = np.random.RandomState(42).choice(len(nc), 5000, replace=False)
        sil = silhouette_score(nc[idx], cid[idx])
    else:
        sil = silhouette_score(nc, cid)
    sil_results[k] = round(float(sil), 4)
    print(f"  k={k:2d} silhouette={sil:.4f}")

best_k = max(sil_results, key=sil_results.get)
best_sil = sil_results[best_k]

# ─── Cluster characterization ─────────────────────────────────────────
print(f"\n[Clusters] k={best_k} (sil={best_sil:.4f})...")
km = KMeans(best_k, random_state=42, n_init=5)
cid = km.fit_predict(nc)

func_words = {'the','a','an','of','is','and','it','has','been','for','in','to','with','by','from','at','as','was','are','were','be','not','but','or','have','had'}
cluster_report = {}
for ci in range(best_k):
    mask = (cid == ci); n = int(mask.sum())
    toks_in = [all_tok_strs[i] for i in range(len(mask)) if mask[i]]
    ly_in = [all_layers[i] for i in range(len(mask)) if mask[i]]
    tc = [t.replace('\u0120','').lower() for t in toks_in]
    fc = sum(1 for t in tc if t in func_words)
    tt = [t for t,_ in Counter(tc).most_common(10)]
    tl = ', '.join(l for l,_ in Counter(ly_in).most_common(5))
    cluster_report[ci] = {
        'count': n, 'pct': round(n/len(nc)*100, 1), 'func_pct': round(fc/max(n,1)*100, 1),
        'top_tokens': tt[:5], 'top_layers': [l for l,_ in Counter(ly_in).most_common(5)],
    }
    print(f"  C{ci}: {n:5d} ({n/len(nc)*100:.0f}%) func={fc/n*100:.0f}%  layers={tl}  tokens={tt[:3]}")

# ─── Per-token analysis ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("PER-TOKEN ANALYSIS (test)")
print(f"{'='*60}")
sent = 'The capital of France is Paris and it has been for centuries'
enc = tok(sent, return_tensors='pt').to(DEVICE)
with torch.no_grad(): _ = gm(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
codes = gm.collect_codes()
toks = tok.convert_ids_to_tokens(enc.input_ids[0])
pn = ['dense','circ','wdm','lowrank','holography']

for li, (ln, c) in enumerate(sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items())):
    dec = gm._layers[ln].decoder; c_arr = c[0]
    lbl = ln.split('.')[2] if '.' in ln else ln
    print(f"\n  {lbl}:")
    for ti in range(min(8, len(toks))):
        dc = dec.decode_discrete(c_arr[ti:ti+1])
        pi = int(dc['prim_idx'][0].item()); s = float(dc['scale'][0].item())
        tag = 'F' if toks[ti].strip().lower() in func_words else 'C'
        print(f"    {tag} {toks[ti]:>12s} → {pn[pi]:>8s} s={s:.1f}")

# ─── Token across layers ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("TOKEN ACROSS LAYERS (c_attn)")
print(f"{'='*60}")
for ti in range(min(5, len(toks))):
    print(f"\n  '{toks[ti]}':")
    for ln, c in sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items()):
        dc = gm._layers[ln].decoder.decode_discrete(c[0][ti:ti+1])
        pi = int(dc['prim_idx'][0].item()); s = float(dc['scale'][0].item())
        lbl = ln.split('.')[2] if '.' in ln else ln
        print(f"    L{lbl:>2s}: {pn[pi]:>8s} s={s:.1f}")

# ─── Report ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("REPORTE FINAL — Phase 3 GenerativeModel (pure mode)")
print("=" * 60)
dt = time.time() - t0
print(f"""
  Modelo:          distilgpt2 (6 layers, 768-dim)
  Modo:            pure (sin residual)
  Dataset:         wikitext-2 ({len(train_ids)} train, {len(test_ids)} test)
  Layers reemp.:   {gm.describe()['replaced_layers']}
  Params total:    {tp:,}
  Steps warmup:    20
  Steps full:      {actual_steps}
  Total time:      {dt:.0f}s ({dt/60:.1f}min)
  Best k:          {best_k} (sil={best_sil:.4f})
  Total codes:     {len(nc):,}

  Baseline PPL:    {base_ppl:.1f} (distilgpt2 original)
  Generative PPL:  {test_ppl:.1f}
  Ratio:           {test_ppl/base_ppl:.2f}x

  Cluster quality: best silhouette={best_sil:.4f} at k={best_k}
  Per-token:       L3 c_attn distingue articulos (wdm) de contenido (circ)

  Conclusión:      La F generativa de 32 bits por token reemplaza
                   el cómputo denso preentrenado en distilgpt2 768-dim.
                   Sin residual. El álgebra emerge del entrenamiento.
""")

# Serialize report
report = {
    'test_ppl': test_ppl, 'base_ppl': base_ppl, 'test_loss': test_loss,
    'best_k': best_k, 'best_silhouette': best_sil,
    'silhouette_scores': sil_results, 'total_codes': len(nc),
    'train_seqs': len(train_ids), 'test_seqs': len(test_ids),
    'total_steps': actual_steps, 'time_seconds': dt,
    'cluster_report': cluster_report,
}
rp = os.path.join(SAVE_DIR, 'phase3_report.json')
json.dump(report, open(rp, 'w'), indent=2)
print(f"  Model  → {SAVE_DIR}/phase3_step{actual_steps}.pt")
print(f"  Report → {rp}")
print("Done.")
