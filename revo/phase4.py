from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from revo.hyperbolic import project_to_ball


def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]; labels: [B, T]
    logp = torch.log_softmax(logits, dim=-1)
    gathered = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered  # [B, T]


@torch.no_grad()
def compute_nll(model: nn.Module, tokenizer, texts: List[str], max_length: int = 128) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
        loss = float(out.loss.item())
        n_tok = int(enc.input_ids.numel())
        total_loss += loss * n_tok
        total_tokens += n_tok
    mean_nll = float(total_loss / max(1, total_tokens))
    return mean_nll, total_tokens


def _build_unigram_prior(tokenizer, texts: List[str], max_length: int = 128) -> torch.Tensor:
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab <= 0:
        # fallback to GPT-2 default
        vocab = 50257
    counts = torch.ones(vocab, dtype=torch.float32)  # Laplace smoothing
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.view(-1)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    probs = counts / counts.sum()
    return probs  # [V]


@torch.no_grad()
def solomonoff_mixed_nll(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    gamma: float = 0.1,
    max_length: int = 128,
) -> float:
    """Mixture prior: p_mix = (1-gamma)*p_model + gamma*p_prior(unigram).
    Returns mean NLL under the mixed distribution.
    """
    model.eval()
    prior = _build_unigram_prior(tokenizer, texts, max_length=max_length)  # [V]
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        logits = out.logits  # [B, T, V]
        pm = torch.softmax(logits, dim=-1)  # [B, T, V]
        pmix = (1.0 - float(gamma)) * pm + float(gamma) * prior.view(1, 1, -1)
        # Avoid log(0)
        pmix = pmix.clamp_min(1e-12)
        # labels are next-token (teacher forcing)
        labels = enc.input_ids
        loss_bt = -torch.log(pmix.gather(-1, labels.unsqueeze(-1)).squeeze(-1))  # [B, T]
        loss = float(loss_bt.mean().item())
        n_tok = int(labels.numel())
        total_loss += loss * n_tok
        total_tokens += n_tok
    return float(total_loss / max(1, total_tokens))


@torch.no_grad()
def compositional_consistency(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
) -> float:
    """Cosine similarity between last-step logits for text vs token-reversed(text)."""
    def _last_logits_for(ids: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = out.logits  # [B, T, V]
        return logits[:, -1, :].detach().to(dtype=torch.float32)

    model.eval()
    outs_base: List[torch.Tensor] = []
    outs_rev: List[torch.Tensor] = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids
        rev = torch.flip(ids, dims=[1])
        outs_base.append(_last_logits_for(ids))
        outs_rev.append(_last_logits_for(rev))
    A = torch.cat(outs_base, dim=0)
    B = torch.cat(outs_rev, dim=0)
    A = A / (A.norm(dim=1, keepdim=True) + 1e-9)
    B = B / (B.norm(dim=1, keepdim=True) + 1e-9)
    return float((A * B).sum(dim=1).mean().item())


@torch.no_grad()
def hyperbolic_profile(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
) -> Dict[str, float]:
    """Profile last hidden state norms in Euclidean vs projected hyperbolic ball."""
    model.eval()
    e_norms: List[torch.Tensor] = []
    h_norms: List[torch.Tensor] = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :].detach().to(dtype=torch.float32)  # [B, D]
        e = last_hidden.norm(dim=-1)
        z = project_to_ball(last_hidden, c=1.0)
        h = z.norm(dim=-1)
        e_norms.append(e)
        h_norms.append(h)
    e_cat = torch.cat(e_norms, dim=0)
    h_cat = torch.cat(h_norms, dim=0)
    e_mean = float(e_cat.mean().item())
    h_mean = float(h_cat.mean().item())
    ratio = float(h_mean / (e_mean + 1e-9))
    return {"euclid_mean": e_mean, "hyper_mean": h_mean, "hyper_over_euclid": ratio}


@torch.no_grad()
def mdl_surrogate_nll(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    mdl_lambda: float = 0.01,
    max_length: int = 128,
) -> float:
    """MDL surrogate: mean NLL + mdl_lambda * mean token length.

    Interprets length as a simple proxy for description/program size. This is a
    diagnostic metric; it does not change the model, only reports score.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
        loss = float(out.loss.item())
        n_tok = int(enc.input_ids.numel())
        total_loss += loss * n_tok
        total_tokens += n_tok
    mean_nll = float(total_loss / max(1, total_tokens))
    mean_len = float(total_tokens / max(1, len(texts)))
    return float(mean_nll + float(mdl_lambda) * mean_len)
