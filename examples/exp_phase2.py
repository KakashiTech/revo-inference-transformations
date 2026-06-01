"""Phase 2: descongelar atención para diversificar códigos por token."""
import time, math, torch; torch.set_num_threads(8)
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeLaw, StructureDecoder
from revo.generative_law import _DensePrimitive, _CirculantPrimitive, _WDMPrimitive, _LowRankPrimitive, _HolographyPrimitive
from revo._utils import set_by_name
import numpy as np
from sklearn.cluster import KMeans; from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
DEVICE = torch.device('cpu')

class GenerativeResidualLayer(nn.Module):
    def __init__(self, orig_layer, weight, bias, enabled=None, law_hidden=64):
        super().__init__()
        self.orig = orig_layer; out_f, in_f = weight.shape
        self.bank = nn.ModuleDict(); self._name_list = []
        for name in (enabled or ['dense','circulant','wdm','lowrank','holography']):
            ok = True
            if name == 'dense': mod = _DensePrimitive(weight, bias)
            elif name == 'circulant': ok = (in_f == out_f); mod = _CirculantPrimitive(weight, bias) if ok else None
            elif name == 'wdm': ok = (in_f == out_f and in_f % 2 == 0); mod = _WDMPrimitive(weight, bias, bands=2) if ok else None
            elif name == 'lowrank': mod = _LowRankPrimitive(weight, bias)
            elif name == 'holography': mod = _HolographyPrimitive(weight, bias)
            else: continue
            if not ok: continue
            self.bank[name] = mod; self._name_list.append(name)
        self.law = GenerativeLaw(in_f, hidden=law_hidden, n_bits=32)
        self.decoder = StructureDecoder(len(self._name_list))
        self._last_code = None; self.eps = nn.Parameter(torch.tensor(0.01))
    def forward(self, x):
        orig_out = self.orig(x); *b, D = x.shape; flat = x.reshape(-1, D)
        code = self.law(flat); self._last_code = code.detach().reshape(*b, 32)
        dec = self.decoder(code); w = F.softmax(dec['prim_logits']/1.0, dim=-1)
        s = dec['scale'].unsqueeze(-1)
        out = torch.zeros(flat.shape[0], orig_out.shape[-1], device=x.device, dtype=x.dtype)
        for ki, nm in enumerate(self._name_list):
            out = out + w[:,ki:ki+1] * self.bank[nm](flat)
        return orig_out + torch.sigmoid(self.eps) * (out * s).reshape(*b, -1)

def replace_with_residual(model):
    from transformers.pytorch_utils import Conv1D as HFConv1D
    layers = {}
    for name, mod in list(model.named_modules()):
        if 'lm_head' in name or not isinstance(mod, (nn.Linear, HFConv1D)): continue
        if not any(p in name for p in ['c_attn','c_proj','c_fc']): continue
        if HFConv1D and isinstance(mod, HFConv1D):
            W, b = mod.weight.data.T.contiguous(), mod.bias.data if mod.bias is not None else None
        else:
            W, b = mod.weight.data, mod.bias.data if mod.bias is not None else None
        nl = GenerativeResidualLayer(mod, W, b)
        set_by_name(model, name, nl); layers[name] = nl
    return layers

print("Data..."); tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
ds = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train').filter(lambda ex: len(ex['text'].strip())>20)
all_ids = []
for i in range(30):
    ex = ds[i]; enc = tok(ex['text'][:128],truncation=True,max_length=64,padding='max_length',return_tensors='pt')
    all_ids.append(enc.input_ids[0])
ids = torch.stack(all_ids).to(DEVICE); print(f"  {len(ids)} seqs")

print("Model..."); model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
for p in model.parameters(): p.requires_grad = False
layers = replace_with_residual(model)
law_decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  {len(layers)} layers, {law_decoder_params:,} trainable (law+decoder)")

# Phase 1: train law+decoder
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3)
for step in range(1, 31):
    model.train(); perm = torch.randperm(len(ids))
    for start in range(0, len(ids), 4):
        if start//4 >= 2: break
        b = ids[perm[start:start+4]]
        opt.zero_grad()
        out = model(input_ids=b, attention_mask=b)
        F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1)).backward()
        opt.step()
    if step%15==0:
        model.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, 24, 4):
                b = ids[s:s+4]; o = model(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1), reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
        print(f"  P1 step {step:2d} ppl={math.exp(nll/max(nt,1)):.0f}")

# Phase 2: unfreeze attention
print("\nUnfreezing c_attn + c_proj...")
attn_p = 0
for name, layer in layers.items():
    if 'c_attn' in name or 'c_proj' in name:
        for p in layer.orig.parameters(): p.requires_grad = True; attn_p += p.numel()
total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  attention: {attn_p:,}, total: {total_p:,}")

opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
for step in range(1, 101):
    model.train(); perm = torch.randperm(len(ids))
    for start in range(0, len(ids), 4):
        if start//4 >= 2: break
        b = ids[perm[start:start+4]]
        opt2.zero_grad()
        out = model(input_ids=b, attention_mask=b)
        F.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b[:,1:].reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        opt2.step()
    if step%25==0:
        model.eval()
        with torch.no_grad():
            nll, nt = 0., 0
            for s in range(0, 24, 4):
                b = ids[s:s+4]; o = model(input_ids=b, attention_mask=b)
                l = F.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b[:,1:].reshape(-1), reduction='sum')
                nll += l.item(); nt += b[:,1:].numel()
        print(f"  P2 step {step:2d} ppl={math.exp(nll/max(nt,1)):.0f}")

# Collect + cluster
print("\nCollecting..."); model.eval(); ac, at, al = [], [], []
with torch.no_grad():
    for s in range(0, len(ids), 4):
        b = ids[s:s+4]; _ = model(input_ids=b, attention_mask=b)
        for ln, lr in layers.items():
            c = lr._last_code
            if c is None: continue
            dec = tok.convert_ids_to_tokens(b[0].tolist())
            for ti in range(c.shape[1]):
                ac.append(c[0,ti].cpu().numpy()); at.append(dec[ti]); al.append(ln)
cn = np.array(ac); print(f"  {len(cn)} codes")

print("Clustering...")
for k in [6,10,12,16]:
    lb = KMeans(k,random_state=42,n_init=3).fit_predict(cn)
    sil = silhouette_score(cn[:2000],lb[:2000],random_state=42) if len(set(lb))>1 else -1
    print(f"  k={k:2d} sil={sil:.3f}")

k=12; km = KMeans(k,random_state=42,n_init=5); cid = km.fit_predict(cn)
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
    print(f"    top5 tokens: {[t for t,_ in Counter(tl).most_common(5)]}")

# Per-token per-layer analysis
print(f"\n{'='*60}\nPER-TOKEN PER-LAYER (c_attn)\n{'='*60}")
sent = 'The capital of France is Paris and it has been for centuries'
enc = tok(sent, return_tensors='pt').to(DEVICE)
with torch.no_grad(): _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
toks = tok.convert_ids_to_tokens(enc.input_ids[0])
pn = ['dense','circ','wdm','lowrank','holography']

for li, (ln, lr) in enumerate(sorted({n:l for n,l in layers.items() if 'c_attn' in n}.items())):
    c = lr._last_code[0]
    print(f"\n  {ln}:")
    for ti in range(min(8,len(toks))):
        dc = lr.decoder.decode_discrete(c[ti:ti+1])
        pi = dc['prim_idx'][0].item(); s = dc['scale'][0].item()
        tag = 'F' if toks[ti].strip().lower() in func_w else 'C'
        print(f"    {tag} {toks[ti]:>10s} → {pn[pi]:>8s} s={s:.1f}")

print(f"\nDone.")
