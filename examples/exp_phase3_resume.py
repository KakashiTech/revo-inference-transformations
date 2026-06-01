"""Resume Phase 3 training from step25 — limpio, checkpoints cada 25 steps."""
import time, math, torch, json, os, signal
torch.set_num_threads(8)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

DEVICE = torch.device('cpu')
SAVE_DIR = '/home/tuffhk/Work/proyecto REVO/checkpoints/phase3'
os.makedirs(SAVE_DIR, exist_ok=True)
interrupted = False

def handler(signum, frame):
    global interrupted
    if not interrupted:
        print(f"\n[SIGINT] Saving checkpoint...", flush=True)
        interrupted = True
signal.signal(signal.SIGINT, handler)

print("=" * 60, flush=True)
print("PHASE 3 RESUME: step25 → 120", flush=True)
print("=" * 60, flush=True)

# ─── Data ──────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
def load(split, n):
    ds = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split)
    ds = ds.filter(lambda ex: len(ex['text'].strip())>20)
    ids = []
    for i,ex in enumerate(ds):
        if i>=n: break
        enc = tok(ex['text'][:256],truncation=True,max_length=128,padding='max_length',return_tensors='pt')
        ids.append(enc.input_ids[0])
    return torch.stack(ids).to(DEVICE)

train_ids = load('train',100); test_ids = load('test',50)
print(f"Data: {len(train_ids)} train, {len(test_ids)} test", flush=True)

# ─── Model ─────────────────────────────────────────────────────────────
print("Building model...", flush=True)
model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
gm = GenerativeModel(model, enabled=['dense','circulant','wdm','lowrank','holography'],
                     name_patterns=['c_attn','c_proj','c_fc'], mode='pure')

cp = torch.load(f'{SAVE_DIR}/phase3_step25.pt', map_location=DEVICE, weights_only=True)
gm.load_state_dict({'model.'+k: v for k,v in cp['model_state'].items()}, strict=True)
for p in model.parameters(): p.requires_grad = True
tp = sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f"Loaded step {cp['step']}, {tp:,} params", flush=True)

# ─── Train ─────────────────────────────────────────────────────────────
opt = torch.optim.AdamW(gm.parameters(), lr=3e-4)
t0 = time.time()

for step in range(26, 121):
    if interrupted: break
    gm.train(); perm = torch.randperm(len(train_ids)); tl=0; bc=0
    for s in range(0, len(train_ids), 8):
        if interrupted: break
        b = train_ids[perm[s:s+8]]
        attn = (b != tok.pad_token_id).long()
        opt.zero_grad()
        out = gm(input_ids=b, attention_mask=attn)
        logits = out.logits[:,:-1]; labels = b[:,1:]
        mask = (labels != tok.pad_token_id)
        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none')
        loss = (loss.reshape_as(labels)*mask.float()).sum()/mask.sum().clamp(min=1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(gm.parameters(),1.0)
        opt.step(); tl+=loss.item(); bc+=1

    if step % 25 == 0 or step == 120:
        # Save
        torch.save({'model_state': model.state_dict(), 'step': step},
                   f'{SAVE_DIR}/phase3_step{step}.pt')
        # Eval
        gm.eval()
        with torch.no_grad():
            nll,nt=0.,0
            for s in range(0, min(40, len(train_ids)), 8):
                b=train_ids[s:s+8]; attn=(b!=tok.pad_token_id).long()
                o=gm(input_ids=b,attention_mask=attn)
                logits=o.logits[:,:-1];labels=b[:,1:];mask=(labels!=tok.pad_token_id)
                lv=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none')
                nll+=(lv.reshape_as(labels)*mask.float()).sum().item();nt+=mask.sum().item()
            ppl=math.exp(nll/max(nt,1))
        dt=time.time()-t0
        print(f"  step {step:3d} loss={tl/bc:.3f} ppl~{ppl:.0f} ({dt:.0f}s)", flush=True)

# ─── Final eval ────────────────────────────────────────────────────────
actual = step
print(f"\n{'='*60}", flush=True)
print(f"EVAL FINAL step {actual}", flush=True)
print(f"{'='*60}", flush=True)
gm.eval()

base = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE).eval()
def eval_ppl(m, data):
    nll,nt=0.,0
    with torch.no_grad():
        for s in range(0, len(data), 8):
            b=data[s:s+8]; attn=(b!=tok.pad_token_id).long()
            o=m(input_ids=b,attention_mask=attn)
            logits=o.logits[:,:-1];labels=b[:,1:];mask=(labels!=tok.pad_token_id)
            lv=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none')
            nll+=(lv.reshape_as(labels)*mask.float()).sum().item();nt+=mask.sum().item()
    return math.exp(nll/max(nt,1)), nll/max(nt,1)
test_ppl, test_loss = eval_ppl(gm, test_ids)
base_ppl, _ = eval_ppl(base, test_ids)
train_ppl, _ = eval_ppl(gm, train_ids[:40])
print(f"  Baseline:  {base_ppl:.1f}", flush=True)
print(f"  Generative: {test_ppl:.1f} (ratio {test_ppl/base_ppl:.2f}x)", flush=True)
print(f"  Train:     {train_ppl:.1f}", flush=True)

# ─── Codes ─────────────────────────────────────────────────────────────
print("\nCollecting codes...", flush=True)
ac,al,ats=[],[],[]
with torch.no_grad():
    for s in range(0, len(test_ids), 8):
        b=test_ids[s:s+8]; attn=(b!=tok.pad_token_id).long()
        _=gm(input_ids=b,attention_mask=attn)
        for ln,c in gm.collect_codes().items():
            for bi in range(b.shape[0]):
                toks=tok.convert_ids_to_tokens(b[bi].tolist())
                for ti in range(c.shape[1]):
                    ac.append(c[bi,ti].cpu().numpy()); al.append(ln); ats.append(toks[ti])
nc=np.array(ac); print(f"  {len(nc)} codes", flush=True)

# ─── Silhouette ────────────────────────────────────────────────────────
print("Silhouette...", flush=True)
sil={}
for k in [4,6,8,10,12,16,20,24]:
    if k>=len(nc): break
    km=KMeans(k,random_state=42,n_init=3); cid=km.fit_predict(nc)
    idx=np.random.RandomState(42).choice(len(nc),5000,replace=False) if len(nc)>5000 else None
    sil[k]=round(float(silhouette_score(nc[idx] if idx else nc,cid[idx] if idx else cid)),4)
    print(f"  k={k:2d} sil={sil[k]:.4f}", flush=True)
bk=max(sil,key=sil.get); bs=sil[bk]

# ─── Clusters ──────────────────────────────────────────────────────────
print(f"Clusters k={bk}...", flush=True)
km=KMeans(bk,random_state=42,n_init=5); cid=km.fit_predict(nc)
fw={'the','a','an','of','is','and','it','has','been','for','in','to','with','by','from','at','as','was','are','were','be','not','but','or','have','had'}
cr={}
for ci in range(bk):
    m=(cid==ci); n=int(m.sum())
    ti=[ats[i] for i in range(len(m)) if m[i]]; li=[al[i] for i in range(len(m)) if m[i]]
    tc=[t.replace('\u0120','').lower() for t in ti]
    fc=sum(1 for t in tc if t in fw)
    tt=[t for t,_ in Counter(tc).most_common(5)]
    cr[ci]={'count':n,'pct':round(n/len(nc)*100,1),'func_pct':round(fc/max(n,1)*100,1),'top_tokens':tt,'top_layers':[l for l,_ in Counter(li).most_common(3)]}
    print(f"  C{ci}: {n:4d} ({n/len(nc)*100:.0f}%) func={fc/n*100:.0f}% top={tt[:3]}", flush=True)

# ─── Per-token ─────────────────────────────────────────────────────────
print("\nPER-TOKEN (c_attn)", flush=True)
pn=['dense','circ','wdm','lowrank','holography']
for sent in ['The capital of France is Paris and it has been for centuries']:
    enc=tok(sent,return_tensors='pt').to(DEVICE); attn=(enc.input_ids!=tok.pad_token_id).long()
    with torch.no_grad(): _=gm(input_ids=enc.input_ids,attention_mask=attn)
    codes=gm.collect_codes(); toks=tok.convert_ids_to_tokens(enc.input_ids[0])
    print(f"\n  '{sent}':")
    for li,(ln,c) in enumerate(sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items())):
        if li>3: break
        lbl=ln.split('.')[2] if '.' in ln else ln
        line=f"    {lbl}: "
        for ti in range(min(8,len(toks))):
            dc=gm._layers[ln].decoder.decode_discrete(c[0][ti:ti+1])
            pi=int(dc['prim_idx'][0].item());s=float(dc['scale'][0].item())
            line+=f"{pn[pi][:3]}{s:.0f} "
        print(line, flush=True)

print("\nTOKEN ACROSS LAYERS", flush=True)
sent='The capital of France is Paris'
enc=tok(sent,return_tensors='pt').to(DEVICE); attn=(enc.input_ids!=tok.pad_token_id).long()
with torch.no_grad(): _=gm(input_ids=enc.input_ids,attention_mask=attn)
codes=gm.collect_codes(); toks=tok.convert_ids_to_tokens(enc.input_ids[0])
for ti in range(min(5,len(toks))):
    print(f"\n  '{toks[ti]}':")
    for ln,c in sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items()):
        dc=gm._layers[ln].decoder.decode_discrete(c[0][ti:ti+1])
        pi=int(dc['prim_idx'][0].item());s=float(dc['scale'][0].item())
        lbl=ln.split('.')[2] if '.' in ln else ln
        print(f"    L{lbl:>2s}: {pn[pi]:>8s} s={s:.1f}", flush=True)

# ─── Report ────────────────────────────────────────────────────────────
dt=time.time()-t0
print("\n"+"="*60, flush=True)
print(f"REPORTE FINAL — Phase 3 (step {actual})", flush=True)
print("="*60, flush=True)
print(f"""
  distilgpt2 768-dim, pure mode, {len(train_ids)} train / {len(test_ids)} test
  {gm.describe()['replaced_layers']} layers, {tp:,} params
  Steps: {actual}  Time: {dt:.0f}s

  Baseline PPL:   {base_ppl:.1f}
  Generative PPL: {test_ppl:.1f}  ({test_ppl/base_ppl:.2f}x baseline)
  Train PPL:      {train_ppl:.1f}

  Silhouette: bk={bk} bs={bs:.4f}  Codes: {len(nc):,}
""", flush=True)

json.dump({'steps':actual,'test_ppl':test_ppl,'base_ppl':base_ppl,'train_ppl':train_ppl,
           'best_k':bk,'best_silhouette':bs,'silhouette_scores':sil,'total_codes':len(nc),
           'cluster_report':cr,'time_seconds':dt},
          open(f'{SAVE_DIR}/phase3_report.json','w'),indent=2)
print(f"Report → {SAVE_DIR}/phase3_report.json", flush=True)
print("Done.", flush=True)
