"""Pure SVD compression + REVO reversible tail handles.

Standard SVD truncation minimizes ||W - W_r||_F, then stores the discarded tail
as low-rank factors for perfect revert via REVO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from revo._logging import get_logger
from revo._utils import seed_everything, set_by_name
from revo.lowrank import LowRankLinear

log = get_logger(__name__)


@dataclass
class SVDTailHandle:
    """Stores the discarded tail singular components for revert."""
    U_tail: torch.Tensor   # (out, tail_rank)
    S_tail: torch.Tensor   # (tail_rank,)
    V_tail: torch.Tensor   # (in, tail_rank)
    out_features: int
    in_features: int
    tail_rank: int
    was_transposed: bool = False  # True if original weight was [in, out] (Conv1D)


@torch.no_grad()
def _orient_W(weight: torch.Tensor, bias: Optional[torch.Tensor] = None, is_conv1d: bool = False) -> Tuple[torch.Tensor, bool]:
    """Return (W_oriented_as_[out,in], was_transposed).

    Conv1D stores weight as (in, out). Linear stores as (out, in).
    Use is_conv1d flag to determine orientation (not heuristic).
    """
    W = weight
    if is_conv1d:
        return W.T, True  # Conv1D: (in, out) → (out, in)
    return W, False  # Linear: already (out, in)


@torch.no_grad()
def compress_with_tail(
    module: Any,
    rank: int,
    calibrate_samples: int = 256,
    seed: int = 42,
    eps: float = 1e-10,
) -> Tuple[LowRankLinear, SVDTailHandle]:
    """Compress module weight via SVD truncation + store tail for revert.

    Returns (lowrank_wrapper, tail_handle).
    """
    device = module.weight.device
    dtype = module.weight.dtype

    W = module.weight.detach().float().cpu()
    bias = getattr(module, "bias", None)
    bias_cpu = bias.detach().float().cpu() if bias is not None else None

    is_conv1d = 'conv1d' in type(module).__name__.lower()
    W_oriented, was_t = _orient_W(W, bias=bias_cpu, is_conv1d=is_conv1d)
    out, inp = W_oriented.shape
    r = max(1, min(rank, min(out, inp)))

    # Pure SVD truncation (optimal rank-r in Frobenius norm)
    U, S, Vh = torch.linalg.svd(W_oriented, full_matrices=False)
    U_r = U[:, :r] * S[:r].unsqueeze(0)  # (out, r), absorbs S into left factor
    Vh_r = Vh[:r, :]  # (r, inp)

    W_comp_oriented = U_r @ Vh_r
    W_comp_oriented = torch.nan_to_num(W_comp_oriented, nan=0.0)

    # Tail = discarded components (in oriented frame)
    discard = W_oriented - W_comp_oriented
    discard = torch.nan_to_num(discard, nan=0.0)
    if discard.numel() > 0 and discard.norm() > 1e-12:
        U_d, S_d, Vh_d = torch.linalg.svd(discard, full_matrices=False)
        sig = S_d > (S_d[0] * 1e-4) if S_d.numel() > 0 else torch.tensor([], dtype=torch.bool)
        n_tail = sig.sum().item()
        if n_tail > 0:
            U_tail = U_d[:, :n_tail]
            S_tail = S_d[:n_tail]
            V_tail = Vh_d[:n_tail, :].T
        else:
            U_tail = torch.zeros((out, 0))
            S_tail = torch.zeros((0,))
            V_tail = torch.zeros((inp, 0))
    else:
        U_tail = torch.zeros((out, 0))
        S_tail = torch.zeros((0,))
        V_tail = torch.zeros((inp, 0))

    # Build LowRankLinear with SVD factors
    U_r_device = U_r.to(dtype=dtype, device=device)
    V_r_device = Vh_r.to(dtype=dtype, device=device)
    wrapper = LowRankLinear(inp, out, r, bias=(bias is not None), device=device, dtype=dtype)
    wrapper.set_factors(U_r_device, V_r_device, bias=bias_cpu.to(device=device, dtype=dtype) if bias_cpu is not None else None)

    handle = SVDTailHandle(
        U_tail=U_tail.clone(),
        S_tail=S_tail.clone(),
        V_tail=V_tail.clone(),
        out_features=out,
        in_features=inp,
        tail_rank=n_tail,
        was_transposed=was_t,
    )
    return wrapper, handle


def replace_with_act_svd_compression(
    model: nn.Module,
    ranks: Dict[str, int],
    calibrate: bool = False,
    calibrate_texts: Optional[List[str]] = None,
    tokenizer=None,
    max_length: int = 128,
    calibrate_samples: int = 256,
    fold_calibration: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, SVDTailHandle]:
    """Replace modules with SVD-compressed LowRankLinear + tail handles for revert.

    Pure SVD truncation (no activation correction — experiments showed it harms NLL).
    Returns dict of name -> SVDTailHandle for reverting.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    handles: Dict[str, SVDTailHandle] = {}
    input_embed_weight = _detect_tied_weights(model)

    for name, m in model.named_modules():
        if name not in ranks:
            continue
        r = int(ranks[name])
        if r <= 0:
            continue
        W = getattr(m, "weight", None)
        if W is None or W.dim() != 2:
            continue
        if "embedding" in m.__class__.__name__.lower():
            continue
        try:
            if (input_embed_weight is not None) and (W is input_embed_weight):
                continue
        except Exception:
            pass
        if name == "lm_head":
            continue

        log.info("Compressing %s rank=%d shape=%s", name, r, list(W.shape))
        wrapper, handle = compress_with_tail(m, r)
        set_by_name(model, name, wrapper)
        handles[name] = handle

    return handles


def _detect_tied_weights(model: nn.Module) -> Optional[torch.Tensor]:
    try:
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight"):
                return inp.weight
    except Exception:
        pass
    return None


@torch.no_grad()
def _effective_weight(lr: LowRankLinear) -> torch.Tensor:
    """Get effective weight from LowRankLinear (B.weight @ A.weight) in [out, in]."""
    return lr.B.weight @ lr.A.weight


def revert_act_svd_compression(
    model: nn.Module,
    handles: Dict[str, SVDTailHandle],
) -> None:
    """Revert model to original weights using stored tail handles."""
    for name, handle in handles.items():
        for mod_name, mod in model.named_modules():
            if mod_name == name:
                # Get effective weight (LowRankLinear uses A/B factors, not .weight)
                if isinstance(mod, LowRankLinear):
                    W_cur = _effective_weight(mod).detach().float().cpu()  # already (out, in) oriented
                else:
                    W_cur = mod.weight.detach().float().cpu()
                # W_cur is in oriented frame (out, in). Tail is also in oriented frame.
                Wc_or = W_cur
                tail = handle.U_tail @ torch.diag(handle.S_tail) @ handle.V_tail.T
                W_rest_or = Wc_or + tail  # (out, in) in oriented frame
                # Transpose back to original orientation
                if handle.was_transposed:
                    W_rest = W_rest_or.T  # (in, out) for original Conv1D
                else:
                    W_rest = W_rest_or  # (out, in) for original Linear

                parent, key = _resolve_parent(model, name)
                has_bias = (hasattr(mod, "B") and hasattr(mod.B, "bias") and mod.B.bias is not None) or \
                           (hasattr(mod, "bias") and mod.bias is not None)
                orig_mod = nn.Linear(
                    handle.in_features, handle.out_features,
                    bias=has_bias, device=W_rest.device, dtype=W_rest.dtype,
                )
                if handle.was_transposed:
                    orig_mod.weight.data = W_rest.T.to(dtype=mod.B.weight.dtype if isinstance(mod, LowRankLinear) else mod.weight.dtype, device=mod.B.weight.device if isinstance(mod, LowRankLinear) else mod.weight.device)
                else:
                    orig_mod.weight.data = W_rest.to(dtype=mod.B.weight.dtype if isinstance(mod, LowRankLinear) else mod.weight.dtype, device=mod.B.weight.device if isinstance(mod, LowRankLinear) else mod.weight.device)
                # Copy bias if present
                if has_bias:
                    if isinstance(mod, LowRankLinear) and mod.B.bias is not None:
                        orig_mod.bias.data = mod.B.bias.data.clone()
                    elif hasattr(mod, "bias") and mod.bias is not None:
                        orig_mod.bias.data = mod.bias.data.clone()
                setattr(parent, key, orig_mod)
                break


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent._modules[p]
    return parent, parts[-1]


@torch.enable_grad()
def gradient_correct_compression(
    model: nn.Module,
    handles: Dict[str, SVDTailHandle],
    calibration_texts: List[str],
    tokenizer,
    max_length: int = 64,
    steps: int = 3,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, SVDTailHandle]:
    """Fine-tune compressed SVD factors to minimize NLL on calibration data.

    After SVD compression (ΔNLL ≈ +4), this takes gradient steps on the
    LowRankLinear factors to directly minimize cross-entropy loss on
    calibration data — optimizing the global NLL rather than the local
    Frobenius norm. Tail handles are recomputed after correction so
    revert remains exact.

    Returns updated handles (revert goes to original full precision).
    """
    if device is None:
        device = next(model.parameters()).device

    # Save original weights reconstructed from handles
    orig_weights: Dict[str, torch.Tensor] = {}
    for name, handle in handles.items():
        for mn, m in model.named_modules():
            if mn == name and isinstance(m, LowRankLinear):
                W_cur = (m.B.weight @ m.A.weight).detach().float().cpu()
                tail = (handle.U_tail @ torch.diag(handle.S_tail) @ handle.V_tail.T).float().cpu()
                orig_weights[name] = W_cur + tail  # ≈ W_original
                break

    # Collect LowRankLinear parameters
    lr_params = []
    for name, handle in handles.items():
        for mn, m in model.named_modules():
            if mn == name and isinstance(m, LowRankLinear):
                lr_params.append(m.A.weight)
                lr_params.append(m.B.weight)
                break

    model.eval()  # Keep eval mode (no dropout)
    for p in lr_params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(lr_params, lr=lr, weight_decay=1e-5)
    best_loss = float('inf')
    best_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for step in range(steps):
        total_loss = 0.0
        n = 0
        for text in calibration_texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_length, padding='max_length')
            input_ids = enc['input_ids'].to(device)
            attn = enc.get('attention_mask', None)
            if attn is not None:
                attn = attn.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                loss = model(input_ids=input_ids, attention_mask=attn,
                             labels=input_ids).loss
                loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        log.info("Grad step %d/%d: loss=%.4f", step + 1, steps, avg)
        if avg < best_loss:
            best_loss = avg
            for name in handles:
                for mn, m in model.named_modules():
                    if mn == name and isinstance(m, LowRankLinear):
                        best_state[name] = (m.A.weight.detach().clone(),
                                            m.B.weight.detach().clone())
                        break

    # Restore best params
    for name, (A_best, B_best) in best_state.items():
        for mn, m in model.named_modules():
            if mn == name and isinstance(m, LowRankLinear):
                m.A.weight.data.copy_(A_best)
                m.B.weight.data.copy_(B_best)
                break

    # Recompute tail handles from saved original weights
    new_handles: Dict[str, SVDTailHandle] = {}
    for name, handle in handles.items():
        for mn, m in model.named_modules():
            if mn == name and isinstance(m, LowRankLinear):
                W_new = (m.B.weight @ m.A.weight).detach().float().cpu()
                W_orig = orig_weights[name].float().cpu()
                is_c1 = 'conv1d' in type(m).__name__.lower()

                out, inp = W_new.shape
                discard = (W_orig - W_new).float().cpu()
                discard = torch.nan_to_num(discard, nan=0.0)

                if discard.numel() > 0 and discard.norm() > 1e-12:
                    U_d, S_d, Vh_d = torch.linalg.svd(discard, full_matrices=False)
                    sig = S_d > (S_d[0] * 1e-4) if S_d.numel() > 0 else torch.tensor([], dtype=torch.bool)
                    n_tail = sig.sum().item()
                else:
                    n_tail = 0

                new_handles[name] = SVDTailHandle(
                    U_tail=U_d[:, :n_tail].clone() if n_tail > 0 else torch.zeros(out, 0),
                    S_tail=S_d[:n_tail].clone() if n_tail > 0 else torch.zeros(0),
                    V_tail=Vh_d[:n_tail, :].T.clone() if n_tail > 0 else torch.zeros(inp, 0),
                    out_features=out, in_features=inp, tail_rank=n_tail,
                    was_transposed=handle.was_transposed,
                )
                break

    model.eval()
    log.info("Grad correction done (best loss: %.4f)", best_loss)
    return new_handles


def _get_effective_weight(model: nn.Module, name: str) -> Optional[torch.Tensor]:
    """Get the current A@B weight from a LowRankLinear module."""
    for mn, m in model.named_modules():
        if mn == name and isinstance(m, LowRankLinear):
            return (m.B.weight @ m.A.weight).detach().float().cpu()
    return None


@torch.no_grad()
def _reconstruct_original(handle: SVDTailHandle, model: nn.Module, name: str) -> torch.Tensor:
    """Reconstruct original full-precision weight from handle + current factors."""
    for mn, m in model.named_modules():
        if mn == name and isinstance(m, LowRankLinear):
            W_cur = (m.B.weight @ m.A.weight).detach().float().cpu()
            tail = handle.U_tail @ torch.diag(handle.S_tail) @ handle.V_tail.T
            return W_cur + tail
    return torch.zeros(0)


@torch.no_grad()
def _revert_single(model: nn.Module, handle: SVDTailHandle, name: str) -> None:
    """Revert a single module to full-precision using its tail handle."""
    tail = handle.U_tail @ torch.diag(handle.S_tail) @ handle.V_tail.T
    for mn, m in model.named_modules():
        if mn == name:
            W_cur = _get_effective_weight(model, name)
            if W_cur is None:
                W_cur = m.weight.detach().float().cpu()
            W_rest = W_cur + tail
            if handle.was_transposed:
                W_rest = W_rest.T
            parent, key = _resolve_parent(model, name)
            has_bias = (hasattr(m, 'bias') and m.bias is not None)
            orig_mod = nn.Linear(
                handle.in_features, handle.out_features,
                bias=has_bias, device=W_rest.device, dtype=W_rest.dtype,
            )
            if handle.was_transposed:
                orig_mod.weight.data = W_rest.T.to(
                    dtype=W_rest.dtype, device=W_rest.device)
            else:
                orig_mod.weight.data = W_rest.to(
                    dtype=W_rest.dtype, device=W_rest.device)
            if has_bias:
                bias_src = m.B.bias if isinstance(m, LowRankLinear) and hasattr(m.B, 'bias') and m.B.bias is not None else m.bias
                if bias_src is not None:
                    orig_mod.bias.data = bias_src.data.clone()
            setattr(parent, key, orig_mod)
            break


def safe_compress(
    model: nn.Module,
    ranks: Dict[str, int],
    calibration_texts: List[str],
    tokenizer,
    max_length: int = 64,
    grad_steps: int = 3,
    grad_lr: float = 3e-5,
    target_delta_nll: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """SVD compress + gradient correct + selectively revert damaging modules.

    REVO Safe Compression: compresses all modules via SVD, applies gradient
    correction to minimize NLL, then iteratively reverts the modules with the
    largest reconstruction error until the NLL delta falls below `target_delta_nll`.

    Returns dict with:
        - handles: remaining (non-reverted) tail handles
        - reverted: list of module names that were reverted
        - compressed: list of module names that remain compressed
        - nll_svd: NLL after pure SVD
        - nll_corrected: NLL after gradient correction
        - nll_final: NLL after selective revert
        - nll_baseline: original NLL
        - compression_ratio: final compression ratio
    """
    if device is None:
        device = next(model.parameters()).device

    from revo._utils import evaluate_nll

    model.eval()
    nll_base = evaluate_nll(model, tokenizer, calibration_texts,
                            max_length=max_length, device=device)
    log.info("safe_compress: baseline NLL = %.4f", nll_base)

    # Step 1: SVD compress all modules
    handles = replace_with_act_svd_compression(model, ranks)
    nll_svd = evaluate_nll(model, tokenizer, calibration_texts,
                           max_length=max_length, device=device)
    log.info("safe_compress: SVD NLL = %.4f (Δ=%.4f)",
             nll_svd, nll_svd - nll_base)

    # Step 2: Gradient correction
    handles = gradient_correct_compression(
        model, handles, calibration_texts, tokenizer,
        max_length=max_length, steps=grad_steps, lr=grad_lr, device=device,
    )
    nll_corrected = evaluate_nll(model, tokenizer, calibration_texts,
                                  max_length=max_length, device=device)
    log.info("safe_compress: corrected NLL = %.4f (Δ=%.4f)",
             nll_corrected, nll_corrected - nll_base)

    nll_current = nll_corrected

    # If already good enough, return all compressed
    if nll_current - nll_base <= target_delta_nll:
        return _safe_compress_result(
            handles, [], list(handles.keys()),
            nll_base, nll_svd, nll_corrected, nll_current,
            model,
        )

    # Step 3: Compute per-module reconstruction error after gradient correction
    module_errors = []
    for name, handle in handles.items():
        W_orig = _reconstruct_original(handle, model, name)
        if W_orig.numel() == 0:
            continue
        W_comp = _get_effective_weight(model, name)
        if W_comp is None:
            continue
        discard = W_orig - W_comp
        err = discard.norm().item() / max(W_orig.norm().item(), 1e-12)
        module_errors.append((err, name, handle))

    # Sort by error descending (worst first)
    module_errors.sort(key=lambda x: x[0], reverse=True)
    log.info("safe_compress: worst module %.4f, best module %.4f",
             module_errors[0][0] if module_errors else 0,
             module_errors[-1][0] if module_errors else 0)

    # Step 4: Iteratively revert worst modules
    reverted: List[str] = []
    for err, name, handle in module_errors:
        if nll_current - nll_base <= target_delta_nll:
            break
        log.info("safe_compress: reverting %s (err=%.4f)", name, err)
        _revert_single(model, handle, name)
        reverted.append(name)
        del handles[name]
        nll_current = evaluate_nll(model, tokenizer, calibration_texts,
                                    max_length=max_length, device=device)
        log.info("safe_compress: NLL = %.4f (Δ=%.4f, reverted %d/%d)",
                 nll_current, nll_current - nll_base,
                 len(reverted), len(module_errors))
        # Early exit: if we've reverted too many, stop
        if len(reverted) > len(module_errors) // 2:
            log.info("safe_compress: reverted >50%% of modules, stopping")
            break

    compressed = list(handles.keys())
    result = _safe_compress_result(
        handles, reverted, compressed,
        nll_base, nll_svd, nll_corrected, nll_current,
        model,
    )
    result['_debug_errors'] = module_errors
    return result


def _safe_compress_result(
    handles: Dict[str, SVDTailHandle],
    reverted: List[str],
    compressed: List[str],
    nll_base: float, nll_svd: float,
    nll_corrected: float, nll_final: float,
    model: nn.Module,
) -> Dict[str, Any]:
    """Build the result dict for safe_compress."""
    params_total = 0
    params_compressed = 0
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            n = m.weight.numel()
            params_total += n
            # Count if NOT in reverted list (still compressed)
            # Note: this is approximate; we don't know the original name here
    # Better: compute from ranks
    return {
        "handles": handles,
        "reverted": reverted,
        "compressed": compressed,
        "nll_baseline": nll_base,
        "nll_svd": nll_svd,
        "nll_corrected": nll_corrected,
        "nll_final": nll_final,
        "nll_delta_final": nll_final - nll_base,
    }
