"""Phase 3: batch=8, 16 seqs/step, 500 total, 200 steps (~40 min)."""
import time, math, torch, json, os, signal
torch.set_num_threads(8)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from revo.generative_law import GenerativeModel
import numpy as np

DEVICE='cpu'
SAVE='/home/tuffhk/Work/proyecto REVO/checkpoints/phase3'
os.makedirs(SAVE,exist_ok=True)
interrupted=False; t0=time.time()

def h(s,f):
    global interrupted
    if not interrupted: print('\nSaving...',flush=True); interrupted=True
signal.signal(signal.SIGINT,h)

print('='*60,flush=True)
print('PHASE 3 TRAIN: 500 seqs, 16/step, 200 steps',flush=True)
print('='*60,flush=True)

# Data
tok=AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token=tok.eos_token
ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')
ds=ds.filter(lambda ex: len(ex['text'].strip())>20)
train_ids,test_ids=[],[]
for i,ex in enumerate(ds):
    if i>=550: break
    enc=tok(ex['text'][:256],truncation=True,max_length=128,padding='max_length',return_tensors='pt')
    if i<500: train_ids.append(enc.input_ids[0])
    else: test_ids.append(enc.input_ids[0])
train_ids=torch.stack(train_ids).to(DEVICE)
test_ids=torch.stack(test_ids).to(DEVICE)
print(f'Train: {len(train_ids)}  Test: {len(test_ids)}',flush=True)

# Model
model=AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
gm=GenerativeModel(model,enabled=['dense','circulant','wdm','lowrank','holography'],
                   name_patterns=['c_attn','c_proj','c_fc'],mode='pure')
for p in model.parameters(): p.requires_grad=True
tp=sum(p.numel() for p in gm.parameters() if p.requires_grad)
print(f'{gm.describe()["replaced_layers"]} layers, {tp:,} params',flush=True)

# Eval
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

base=AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE).eval()
base_ppl=eval_ppl(base,test_ids)
print(f'Baseline PPL: {base_ppl:.1f}',flush=True)

# Train
opt=torch.optim.AdamW(gm.parameters(),lr=3e-4)
SEQS_PER_STEP=16
TOTAL=200

for step in range(1,TOTAL+1):
    if interrupted: break
    gm.train(); perm=torch.randperm(len(train_ids)); total_loss=0; batches=0
    for s in range(0,SEQS_PER_STEP,8):
        if interrupted: break
        b=train_ids[perm[s:s+8]]
        attn=(b!=tok.pad_token_id).long()
        opt.zero_grad()
        out=gm(input_ids=b,attention_mask=attn)
        logits=out.logits[:,:-1];labels=b[:,1:]
        mask=(labels!=tok.pad_token_id)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none')
        loss=(loss.reshape_as(labels)*mask.float()).sum()/mask.sum().clamp(min=1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(gm.parameters(),1.0)
        opt.step(); total_loss+=loss.item(); batches+=1
    if step%50==0 or step==TOTAL:
        gm.eval()
        test_ppl=eval_ppl(gm,test_ids)
        train_ppl=eval_ppl(gm,train_ids[:100])
        dt=time.time()-t0
        print(f'step {step:3d} loss={total_loss/batches:.3f} train={train_ppl:.0f} test={test_ppl:.0f} t={dt:.0f}s',flush=True)
        torch.save({'state':model.state_dict(),'step':step},f'{SAVE}/phase3_step{step}.pt')

actual=step
gm.eval()
test_ppl=eval_ppl(gm,test_ids)
train_ppl=eval_ppl(gm,train_ids[:100])
dt=time.time()-t0
print(f'\n{"="*60}',flush=True)
print(f'FINAL step {actual}:  test PPL={test_ppl:.1f}  train PPL={train_ppl:.0f}  (baseline={base_ppl:.1f})',flush=True)
print(f'Ratio: {test_ppl/base_ppl:.2f}x',flush=True)
print(f'Time: {dt:.0f}s',flush=True)

# Codes + clustering
print('\nCodes...',flush=True)
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
nc=np.array(ac)
print(f'{len(nc)} codes',flush=True)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for k in [8,12,16,24]:
    if k>=len(nc): break
    km=KMeans(k,random_state=42,n_init=3); cid=km.fit_predict(nc)
    idx=np.random.RandomState(42).choice(len(nc),5000,replace=False)
    print(f'k={k:2d} sil={silhouette_score(nc[idx],cid[idx]):.4f}',flush=True)

json.dump({'steps':actual,'test_ppl':test_ppl,'base_ppl':base_ppl,'train_ppl':train_ppl,
           'total_codes':len(nc),'time_seconds':dt},
          open(f'{SAVE}/phase3_report.json','w'),indent=2)
print(f'\nReport → {SAVE}/phase3_report.json',flush=True)
print('Done.',flush=True)
