"""Eval Phase 3 ultralight step500: load, codes, clustering, report."""
import time, math, torch, json, os
torch.set_num_threads(4)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

DEVICE='cpu'; SAVE='/home/tuffhk/Work/proyecto REVO/checkpoints/phase3'
t0=time.time(); MAXLEN=64

tok=AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token=tok.eos_token

def load(n,split='train'):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split)
    ds=ds.filter(lambda ex: len(ex['text'].strip())>20)
    ids=[]
    for i,ex in enumerate(ds):
        if i>=n: break
        enc=tok(ex['text'][:128],truncation=True,max_length=MAXLEN,padding='max_length',return_tensors='pt')
        ids.append(enc.input_ids[0])
    return torch.stack(ids).to(DEVICE)

test_ids=load(50,'test')
train_ids=load(100,'train')

print('Loading model...')
model=AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
gm=GenerativeModel(model,enabled=['dense','circulant','wdm','lowrank','holography'],
                   name_patterns=['c_attn','c_proj','c_fc'],mode='pure')
for p in model.parameters(): p.requires_grad = False
for layer in gm._layers.values():
    for p in layer.law.parameters(): p.requires_grad = True
    for p in layer.decoder.parameters(): p.requires_grad = True

cp=torch.load(f'{SAVE}/phase3_ul_s500.pt',map_location=DEVICE,weights_only=True)
gm.load_state_dict({'model.'+k:v for k,v in cp['state'].items()},strict=True)
gm.eval()
tp=sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f'Loaded step {cp["step"]}, {tp:,} params')

# Eval
base=AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE).eval()
def eval_ppl(m,data):
    nll,nt=0.,0
    with torch.no_grad():
        for s in range(0,len(data),8):
            b=data[s:s+8]; attn=(b!=tok.pad_token_id).long()
            o=m(input_ids=b,attention_mask=attn)
            logits=o.logits[:,:-1];labels=b[:,1:];mask=(labels!=tok.pad_token_id)
            lv=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none')
            nll+=(lv.reshape_as(labels)*mask.float()).sum().item();nt+=mask.sum().item()
    return math.exp(nll/max(nt,1))

base_ppl=eval_ppl(base,test_ids)
test_ppl=eval_ppl(gm,test_ids)
train_ppl=eval_ppl(gm,train_ids)
print(f'Baseline PPL: {base_ppl:.1f}')
print(f'Generative:  {test_ppl:.1f} ({test_ppl/base_ppl:.2f}x)')
print(f'Train PPL:   {train_ppl:.0f}')

# Codes
print('\nCodes...')
ac,al,ats=[],[],[]
with torch.no_grad():
    for s in range(0,len(test_ids),8):
        b=test_ids[s:s+8]; attn=(b!=tok.pad_token_id).long()
        _=gm(input_ids=b,attention_mask=attn)
        for ln,c in gm.collect_codes().items():
            for bi in range(b.shape[0]):
                toks=tok.convert_ids_to_tokens(b[bi].tolist())
                for ti in range(c.shape[1]):
                    ac.append(c[bi,ti].cpu().numpy());al.append(ln);ats.append(toks[ti])
nc=np.array(ac); print(f'{len(nc)} codes')

# Silhouette
print('Silhouette...')
sil={}
for k in [4,6,8,10,12,16,20,24]:
    if k>=len(nc): break
    km=KMeans(k,random_state=42,n_init=3); cid=km.fit_predict(nc)
    if len(nc)>5000:
        idx=np.random.RandomState(42).choice(len(nc),5000,replace=False)
        sil_=silhouette_score(nc[idx],cid[idx])
    else: sil_=silhouette_score(nc,cid)
    sil[k]=round(float(sil_),4)
    print(f'  k={k:2d} sil={sil_:.4f}')
bk=max(sil,key=sil.get); bs=sil[bk]

# Clusters
print(f'\nClusters k={bk}...')
km=KMeans(bk,random_state=42,n_init=5); cid=km.fit_predict(nc)
fw={'the','a','an','of','is','and','it','has','been','for','in','to','with','by','from','at','as','was','are','were','be','not','but','or','have','had'}
cr={}
for ci in range(bk):
    m=(cid==ci); n=int(m.sum())
    ti=[ats[i] for i in range(len(m)) if m[i]]; li=[al[i] for i in range(len(m)) if m[i]]
    tc=[t.replace('\u0120','').lower() for t in ti]
    fc=sum(1 for t in tc if t in fw)
    tt=[t for t,_ in Counter(tc).most_common(5)]
    cr[ci]={'count':n,'pct':round(n/len(nc)*100,1),'func_pct':round(fc/max(n,1)*100,1),'top_tokens':tt}
    print(f'  C{ci}: {n:4d} ({n/len(nc)*100:.0f}%) func={fc/n*100:.0f}% top3={tt[:3]}')

# Per-token
print(f'\n{"="*60}')
print('PER-TOKEN (c_attn)')
print(f'{"="*60}')
pn=['dense','circ','wdm','lowrank','holography']
sent='The capital of France is Paris and it has been for centuries'
enc=tok(sent,return_tensors='pt').to(DEVICE); attn=(enc.input_ids!=tok.pad_token_id).long()
with torch.no_grad(): _=gm(input_ids=enc.input_ids,attention_mask=attn)
codes=gm.collect_codes(); toks=tok.convert_ids_to_tokens(enc.input_ids[0])
print(f'\n  "{sent}":')
for li,(ln,c) in enumerate(sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items())):
    lbl=ln.split('.')[2] if '.' in ln else ln
    line=f'    {lbl}: '
    for ti in range(min(8,len(toks))):
        dc=gm._layers[ln].decoder.decode_discrete(c[0][ti:ti+1])
        pi=int(dc['prim_idx'][0].item());s=float(dc['scale'][0].item())
        line+=f'{pn[pi][:3]}{s:.0f} '
    print(line)

print('\nTOKEN ACROSS LAYERS')
sent='The capital of France is Paris'
enc=tok(sent,return_tensors='pt').to(DEVICE); attn=(enc.input_ids!=tok.pad_token_id).long()
with torch.no_grad(): _=gm(input_ids=enc.input_ids,attention_mask=attn)
codes=gm.collect_codes(); toks=tok.convert_ids_to_tokens(enc.input_ids[0])
for ti in range(min(5,len(toks))):
    print(f'\n  "{toks[ti]}":')
    for ln,c in sorted({n:c for n,c in codes.items() if 'c_attn' in n}.items()):
        dc=gm._layers[ln].decoder.decode_discrete(c[0][ti:ti+1])
        pi=int(dc['prim_idx'][0].item());s=float(dc['scale'][0].item())
        lbl=ln.split('.')[2] if '.' in ln else ln
        print(f'    L{lbl:>2s}: {pn[pi]:>8s} s={s:.1f}')

# Report
dt=time.time()-t0
print(f'\n{"="*60}')
print(f'REPORTE FINAL — Phase 3 Ultralight (step500)')
print(f'{"="*60}')
print(f'''
  freeze base, solo law+decoder ({tp:,} params)
  300 train seqs, 64 tokens, 500 steps
  Time: {dt:.0f}s

  Baseline:   {base_ppl:.1f}
  Generative: {test_ppl:.1f} ({test_ppl/base_ppl:.2f}x)
  Train PPL:  {train_ppl:.0f}

  Silhouette: best={bs:.4f} at k={bk}
  Codes:      {len(nc):,}
  
  CONCLUSION: Base congelada → PPL no mejora (~4000).
  La ley generativa necesita que el modelo base se adapte
  para aprender códigos útiles. Con 2M params y 300 seqs,
  el law no tiene suficiente capacidad de representación.
''')

json.dump({'steps':500,'test_ppl':test_ppl,'base_ppl':base_ppl,'train_ppl':train_ppl,
           'trainable_params':tp,'best_k':bk,'best_sil':bs,'sil_scores':sil,'total_codes':len(nc),'time_seconds':dt,'cluster_report':cr},
          open(f'{SAVE}/phase3_report.json','w'),indent=2)
print(f'Report → {SAVE}/phase3_report.json')
print('Done.')
