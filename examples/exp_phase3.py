"""Phase 3: GenerativeModel puro — sin residual, ley generativa como cómputo."""
import time, math, torch; torch.set_num_threads(8)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeModel
import numpy as np
from sklearn.cluster import KMeans; from sklearn.metrics import silhouette_score
from collections import Counter

DEVICE = torch.device('cpu')
tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
ds = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train').filter(lambda ex: len(ex['text'].strip())>20)
print("Tokenizing...")
all_ids = []
for i in range(30):
    ex = ds[i]; enc = tok(ex['text'][:128],truncation=True,max_length=64,padding='max_length',return_tensors='pt')
    all_ids.append(enc.input_ids[0])
ids = torch.stack(all_ids).to(DEVICE)

model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
gm = GenerativeModel(model, enabled=['dense','circulant','wdm','lowrank','holography'],
                     name_patterns=['c_attn','c_proj','c_fc'])
# Freeze only original transformer weights (not generative layers)
for n, p in model.named_parameters():
    if 'law' not in n and 'decoder' not in n and 'code_to_logits' not in n:
        p.requires_grad = False
tp = sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f"Warmup {gm.describe()['replaced_layers']} layers, {tp:,} trainable")

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gm.parameters()), lr=3e-3)
for step in range(1, 21):
    gm.train(); perm = torch.randperm(len(ids))
    for start in range(0, len(ids), 4):
        if start//4 >= 2: break
        b = ids[perm[start:start+4]]
        opt.zero_grad()
        out = gm(input_ids=b, attention_mask=b)
        F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1)).backward()
        opt.step()
    if step%10==0:
        gm.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, 24, 4):
                b = ids[s:s+4]; o = gm(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1), reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
        print(f"  warmup {step:2d} ppl={math.exp(nll/max(nt,1)):.0f}")

# Unfreeze everything for full training
for p in model.parameters(): p.requires_grad = True
tp = sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f"\nFull training ({tp:,} params)...")
opt2 = torch.optim.AdamW(gm.parameters(), lr=3e-4)
for step in range(1, 101):
    gm.train(); perm = torch.randperm(len(ids))
    total_loss = 0.0
    for start in range(0, len(ids), 4):
        if start//4 >= 2: break
        b = ids[perm[start:start+4]]
        opt2.zero_grad()
        out = gm(input_ids=b, attention_mask=b)
        loss = F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0)
        opt2.step()
        total_loss += loss.item()
    if step%25==0:
        gm.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, 24, 4):
                b = ids[s:s+4]; o = gm(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1), reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
        print(f"  full {step:3d} loss={total_loss/2:.3f} ppl={math.exp(nll/max(nt,1)):.0f}")

# Collect + analyze
print("\nCollecting..."); gm.eval(); ac, at, al = [], [], []
with torch.no_grad():
    for s in range(0, len(ids), 4):
        b = ids[s:s+4]; _ = gm(input_ids=b, attention_mask=b)
        codes = gm.collect_codes()
        for ln, c in codes.items():
            dec = tok.convert_ids_to_tokens(b[0].tolist())
            for ti in range(c.shape[1]):
                ac.append(c[0,ti].cpu().numpy()); at.append(dec[ti]); al.append(ln)
cn = np.array(ac); print(f"  {len(cn)} codes")

print("Clustering...")
for k in [6,8,10,12,16]:
    lb = KMeans(k,random_state=42,n_init=3).fit_predict(cn)
    sil = silhouette_score(cn[:2000],lb[:2000],random_state=42) if len(set(lb))>1 else -1
    print(f"  k={k:2d} sil={sil:.3f}")

k=10; km = KMeans(k,random_state=42,n_init=5); cid = km.fit_predict(cn)
func_w = {'the','a','an','of','is','and','it','has','been','for','in','to','with','by','from','at','as'}
print(f"\n{'='*60}\nCLUSTERS (k={k})\n{'='*60}")
for ci in range(k):
    m = (cid==ci); n = m.sum()
    tl = [at[i].replace('Ġ','').lower() for i,m2 in enumerate(m) if m2]
    fc = sum(1 for t in tl if t in func_w)
    ly = [al[i] for i,m2 in enumerate(m) if m2]
    def sn(n): p=n.split('.'); return f'h.{p[2]}.{p[3]}' if len(p)>3 else n
    ts = ', '.join(sn(n) for n,_ in Counter(ly).most_common(4))
    print(f"\n  C{ci}: {n:4d} codes ({n/len(cn)*100:.0f}%)  func={fc/n*100:.0f}%")
    print(f"    layers: {ts}")
    print(f"    top5: {[t for t,_ in Counter(tl).most_common(5)]}")

print(f"\n{'='*60}\nPER-TOKEN (c_attn)\n{'='*60}")
sent = 'The capital of France is Paris and it has been for centuries'
enc = tok(sent, return_tensors='pt').to(DEVICE)
with torch.no_grad(): _ = gm(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
codes = gm.collect_codes(); toks = tok.convert_ids_to_tokens(enc.input_ids[0])
pn = ['dense','circ','wdm','lowrank','holography']
for li, (ln, c) in enumerate(sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items())):
    dec = gm._layers[ln].decoder; c = c[0]
    print(f"\n  {ln.split('.')[2]}:")
    for ti in range(min(8,len(toks))):
        dc = dec.decode_discrete(c[ti:ti+1])
        pi = dc['prim_idx'][0].item(); s = dc['scale'][0].item()
        tag = 'F' if toks[ti].strip().lower() in func_w else 'C'
        print(f"    {tag} {toks[ti]:>10s} → {pn[pi]:>8s} s={s:.1f}")

# Same token across all layers
print(f"\n{'='*60}\nTOKEN ACROSS LAYERS")
print(f"{'='*60}")
for ti in range(min(4,len(toks))):
    print(f"\n  '{toks[ti]}':")
    for ln, c in sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items()):
        c_arr = c[0]; dec = gm._layers[ln].decoder
        dc = dec.decode_discrete(c_arr[ti:ti+1])
        pi = dc['prim_idx'][0].item(); s = dc['scale'][0].item()
        print(f"    L{ln.split('.')[2]}: {pn[pi]:>8s} s={s:.1f}")
print("Done.")
