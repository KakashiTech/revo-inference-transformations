from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F


class RadixKVCache:
    """Prefix-tree KV cache for causal LM inference using past_key_values.

    Stores past_key_values keyed by token prefix (tuple[int]).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.store: Dict[Tuple[int, ...], Optional[Tuple]] = {}
        self.store[tuple()] = None
        self.device = device

    def longest_prefix(self, seq: List[int]) -> int:
        # Return length of the longest cached prefix of seq
        for l in range(len(seq), -1, -1):
            if tuple(seq[:l]) in self.store:
                return l
        return 0

    def get(self, prefix: List[int]) -> Optional[Tuple]:
        return self.store.get(tuple(prefix), None)

    def put(self, prefix: List[int], past_kv: Optional[Tuple]) -> None:
        self.store[tuple(prefix)] = past_kv


@torch.no_grad()
def radix_eval_nll(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
) -> Dict[str, float]:
    model.eval()
    device = next(iter(model.parameters())).device
    total_loss = 0.0
    total_tokens = 0
    calls = 0
    cache = RadixKVCache(device=device)
    # Tokenize all
    sequences: List[List[int]] = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids[0].tolist()
        sequences.append(ids)
    # Evaluate per sequence using shared cache
    for ids in sequences:
        L = len(ids)
        if L <= 1:
            continue
        naive_tokens = L - 1
        total_tokens += naive_tokens
        # Iterate positions k=1..L-1 predicting token ids[k]
        for i in range(1, L):
            prefix = ids[:i]
            # Find longest cached prefix <= i-1
            l = cache.longest_prefix(prefix)
            past = cache.get(ids[:l])
            # Advance from length l to i using one-step calls
            for j in range(l, i):
                inp = torch.tensor([[ids[j]]], device=device, dtype=torch.long)
                out = model(input_ids=inp, use_cache=True, past_key_values=past, return_dict=True)
                past = out.past_key_values
                cache.put(ids[: j + 1], past)
                calls += 1
            # Now have logits for next token distribution (after consuming ids[i-1])
            logits = out.logits[:, -1, :]  # [1, V]
            target = torch.tensor([ids[i]], device=device, dtype=torch.long)
            logprob = F.log_softmax(logits, dim=-1).gather(1, target.view(1, 1)).squeeze()
            total_loss += float(-logprob.item())
    mean_nll = float(total_loss / max(1, total_tokens))
    return {
        "mean_nll": mean_nll,
        "tokens": float(total_tokens),
        "calls": float(calls),
        "saved_calls": float(total_tokens - calls),
        "saved_calls_ratio": float((total_tokens - calls) / max(1, total_tokens)),
        "prefix_nodes": float(len(cache.store)),
    }
