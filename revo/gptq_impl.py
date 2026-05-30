import torch
import torch.nn as nn


def _is_quantizable(mod):
    t = type(mod).__name__
    return t in ('Linear', 'Conv1D') and hasattr(mod, 'weight') and mod.weight.dim() == 2


@torch.no_grad()
def _get_hessian(model, module_names, calibration_loader, device='cpu'):
    """Collect input activations and compute Hessian for each module.

    Returns  {name: (H, W, bias)}  where H.shape = (d_in, d_in), W.shape = (d_out, d_in)
    """
    hessians = {}

    def _hook(name):
        def _store(m, inp, _out):
            if inp[0] is None:
                return
            x = inp[0].detach().to(device).float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = x
            else:
                hessians[name] = torch.cat([hessians[name], x], dim=0)
        return _store

    hooks = []
    for name, mod in model.named_modules():
        if name in module_names:
            hooks.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    for texts in calibration_loader:
        _ = model(texts)

    for h in hooks:
        h.remove()

    result = {}
    for name, mod in model.named_modules():
        if name not in module_names:
            continue
        if name not in hessians:
            continue
        W = mod.weight.data.float()
        if type(mod).__name__ == 'Conv1D':
            W = W.T.contiguous()
        bias = mod.bias.data.float() if hasattr(mod, 'bias') and mod.bias is not None else None
        X = hessians[name]
        H = 2.0 * (X.T @ X)
        result[name] = (H, W, bias)
    return result


@torch.no_grad()
def _quantize_group(w_group, bits=4, scale=None):
    """Quantize a group of weights using min-max.

    w_group: (d_out, group_size)
    scale: pre-computed scale (or None to compute from data)
    Returns: (w_quantized, scale)
    """
    n_levels = 2 ** bits
    q_max = n_levels // 2 - 1
    if scale is None:
        w_max = w_group.abs().max()
        scale = w_max / q_max if w_max > 1e-10 else 1.0
    q = (w_group / scale).round().clamp(-q_max, q_max) * scale
    return q, scale


def quantize_minmax(W, bits=4, group_size=128):
    """Simple min-max per-group quantization. Returns W_q and per-group scales."""
    d_out, d_in = W.shape
    n_levels = 2 ** bits
    q_max = n_levels // 2 - 1
    n_groups = (d_in + group_size - 1) // group_size

    W_q = W.clone().float()
    scales = torch.zeros(n_groups, device=W.device)

    for g in range(n_groups):
        gs = g * group_size
        ge = min(gs + group_size, d_in)
        w_max = W_q[:, gs:ge].abs().max()
        scale = w_max / q_max if w_max > 1e-10 else 1.0
        scales[g] = scale
        W_q[:, gs:ge] = (W_q[:, gs:ge] / scale).round().clamp(-q_max, q_max) * scale

    return W_q, scales


def gptq_quantize_layer(module, H, bits=4, group_size=128, damp=0.1):
    """GPTQ: single-pass over ALL columns with pre-computed scales.

    Uses Hessian-based compensation to minimize expected output error.
    """
    W_orig = module.weight.data.clone()
    is_conv1d = type(module).__name__ == 'Conv1D'
    W = W_orig.T.contiguous() if is_conv1d else W_orig
    d_out, d_in = W.shape
    n_levels = 2 ** bits
    q_max = n_levels // 2 - 1
    device = W.device

    # Pre-compute group scales from original W
    n_groups = (d_in + group_size - 1) // group_size
    scales = torch.zeros(n_groups, device=device)
    for g in range(n_groups):
        gs = g * group_size
        ge = min(gs + group_size, d_in)
        w_max = W[:, gs:ge].abs().max()
        scales[g] = w_max / q_max if w_max > 1e-10 else 1.0

    # Inverse Hessian
    H_reg = H.clone().float()
    damp_val = damp * H_reg.diag().mean()
    H_reg.diagonal().add_(damp_val)

    try:
        H_inv = torch.linalg.inv(H_reg)
    except RuntimeError:
        H_reg.diagonal().add_(damp_val * 10)
        H_inv = torch.linalg.inv(H_reg)

    H_inv = H_inv.float().clamp(-100, 100)

    # Order: most important first (largest H_inv diag)
    order = torch.argsort(H_inv.diag(), descending=True)

    # GPTQ main loop
    W_float = W.clone().float()
    W_quant = torch.zeros_like(W_float)

    for idx in range(d_in):
        j = order[idx].item()
        g = j // group_size
        scale = scales[g].item()

        # Quantize column j
        raw = W_float[:, j]
        q_col = (raw / scale).round().clamp(-q_max, q_max) * scale
        err = q_col - raw
        W_quant[:, j] = q_col

        # Compensate remaining columns
        if idx < d_in - 1:
            remaining = order[idx + 1:]
            h_jj = H_inv[j, j].item()
            if abs(h_jj) > 1e-12:
                alphas = (H_inv[j, remaining] / h_jj).unsqueeze(0)
                # Clamp alpha to prevent excessive compensation
                alphas = alphas.clamp(-2.0, 2.0)
                W_float[:, remaining] -= err.unsqueeze(-1) * alphas

    # Final quantization pass with pre-computed scales
    for g in range(n_groups):
        gs = g * group_size
        ge = min(gs + group_size, d_in)
        scale = scales[g].item()
        W_quant[:, gs:ge] = (W_float[:, gs:ge] / scale).round().clamp(-q_max, q_max) * scale

    # Write back
    W_q = W_quant
    if is_conv1d:
        W_q = W_q.T.contiguous()
    module.weight.data = W_q.to(module.weight.dtype)

    return {'W_orig': W_orig, 'bits': bits, 'group_size': group_size}


def gptq_quantize_model(model, calibration_texts, tokenizer, bits=4, group_size=128,
                        max_length=128, device='cpu'):
    module_names = set()
    for name, mod in model.named_modules():
        if _is_quantizable(mod):
            if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                module_names.add(name)

    print(f'GPTQ quantizing {len(module_names)} modules to {bits}-bit (group={group_size})...')

    class _CalibLoader:
        def __init__(self, texts, tok, max_len, dev):
            self.texts, self.tok, self.max_len, self.dev = texts, tok, max_len, dev
        def __iter__(self):
            for t in self.texts:
                enc = self.tok(t, return_tensors='pt', truncation=True, max_length=self.max_len)
                yield enc['input_ids'].to(self.dev)

    loader = _CalibLoader(calibration_texts, tokenizer, max_length, device)

    print('  Collecting Hessians...')
    hessian_data = _get_hessian(model, module_names, loader, device=device)
    print(f'  Got {len(hessian_data)} Hessians')

    handles = {}
    for name in module_names:
        if name not in hessian_data:
            continue
        H, W, bias = hessian_data[name]
        for mn, mod in model.named_modules():
            if mn == name:
                h = gptq_quantize_layer(mod, H, bits=bits, group_size=group_size)
                handles[name] = h
                break

    print(f'  Done. {len(handles)} modules quantized.')
    return handles


def gptq_revert(model, handles):
    for name, h in handles.items():
        for mn, mod in model.named_modules():
            if mn == name and hasattr(mod, 'weight'):
                mod.weight.data.copy_(h['W_orig'])
                break
