"""REVO CLI: unified command-line interface.

Usage:
    revo compress --model <model> --ratio 1.5
    revo compress --model <model> --quantize --bits 4
    revo run --model <model>
"""

from __future__ import annotations

import argparse
import sys

import torch

from revo._utils import load_model_tokenizer, gen_texts, save_results_json, evaluate_nll


def _energy_from_ratio(ratio: float) -> float:
    return max(0.5, min(0.99, 1.0 - (ratio - 1.0) * 0.1))


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)


def register_compress_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("compress", help="Compress model with REVO safe compression")
    _add_common_args(p)
    p.add_argument("--ratio", type=float, default=1.5, help="Target SVD compression ratio")
    p.add_argument("--energy-keep", type=float, default=None,
                   help="Energy threshold (default: auto from --ratio)")
    p.add_argument("--revert-on-delta", type=float, default=0.5,
                   help="Revert modules if NLL delta exceeds this")
    p.add_argument("--grad-steps", type=int, default=3,
                   help="Gradient correction steps")
    p.add_argument("--quantize", action="store_true",
                   help="Apply int4 quantization after SVD compression")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4],
                   help="Quantization bits (default 4)")
    p.add_argument("--group-size", type=int, default=128,
                   help="Quantization group size (default 128)")
    p.set_defaults(func=cmd_compress)


def cmd_compress(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    print(f"╔══ REVO Compress ═══════════════════════════════════")
    print(f"║ Model:  {args.model}")
    print(f"║ Ratio:  {args.ratio}×  (revert Δ < {args.revert_on_delta})")
    if args.quantize:
        print(f"║ Quant:  {args.bits}-bit (group={args.group_size})")

    model, tokenizer = load_model_tokenizer(args.model)
    device = next(model.parameters()).device
    texts = gen_texts(args.prompts)

    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    from revo.act_svd import safe_compress

    energy_keep = args.energy_keep if args.energy_keep is not None else _energy_from_ratio(args.ratio)
    print(f"║ Energy: {energy_keep:.2f}")

    prof = profile_model_2d(model)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep, max_rank=32)
    active_ranks = {k: v for k, v in ranks.items() if v > 0}
    print(f"║ Layers: {len(ranks)} total, {len(active_ranks)} compressible")

    result = safe_compress(
        model, ranks, texts, tokenizer,
        max_length=args.max_length,
        grad_steps=args.grad_steps, grad_lr=3e-5,
        target_delta_nll=args.revert_on_delta,
        device=device,
    )

    nll_delta_svd = result["nll_svd"] - result["nll_baseline"]
    nll_delta_corr = result["nll_corrected"] - result["nll_baseline"]
    print(f"║")
    print(f"╠══ SVD Compression ═════════════════════════════════")
    print(f"║  Baseline NLL:       {result['nll_baseline']:.6f}")
    print(f"║  After SVD:          {result['nll_svd']:.6f}  (Δ={nll_delta_svd:+.6f})")
    print(f"║  After correction:   {result['nll_corrected']:.6f}  (Δ={nll_delta_corr:+.6f})")
    print(f"║  Final NLL:          {result['nll_final']:.6f}  (Δ={result['nll_delta_final']:+.6f})")
    print(f"║  Modules compressed: {len(result['compressed'])}")
    print(f"║  Modules reverted:   {len(result['reverted'])}")

    report = {
        "command": "compress",
        "model": args.model,
        "ratio": args.ratio,
        "energy_keep": energy_keep,
        "revert_on_delta": args.revert_on_delta,
        **{k: v for k, v in result.items() if k != "handles"},
    }

    if args.quantize:
        from revo.gptq_revo import quantize_all, selective_dequantize

        nll_before_q = result["nll_final"]
        print(f"║")
        print(f"╠══ Quantization ══════════════════════════════════")
        print(f"║  Quantizing to {args.bits}-bit (group={args.group_size})...")

        q_handles = quantize_all(
            model, texts, tokenizer,
            max_length=args.max_length,
            bits=args.bits,
            group_size=args.group_size,
            device=device,
        )

        nll_quantized = evaluate_nll(model, tokenizer, texts, max_length=args.max_length)
        nll_delta_q = nll_quantized - nll_before_q
        print(f"║  After quant:        {nll_quantized:.6f}  (Δ={nll_delta_q:+.6f})")

        if nll_delta_q > args.revert_on_delta:
            def eval_fn():
                return evaluate_nll(model, tokenizer, texts, max_length=args.max_length)
            q_handles, reverted_q = selective_dequantize(
                model, q_handles, eval_fn, max_nll_delta=args.revert_on_delta,
            )
            nll_final_q = evaluate_nll(model, tokenizer, texts, max_length=args.max_length)
            print(f"║  After SQ:           {nll_final_q:.6f}  (Δ={nll_final_q - nll_before_q:+.6f})")
            print(f"║  Modules dequant:    {len(reverted_q)}")
            report["quant_reverted"] = reverted_q
            report["nll_after_quant"] = nll_final_q
        else:
            report["nll_after_quant"] = nll_quantized
            report["quant_reverted"] = []

        report["nll_before_quant"] = nll_before_q
        report["quant_modules"] = len(q_handles)
        report["quant_bits"] = args.bits
        report["quant_group_size"] = args.group_size

    print(f"╚════════════════════════════════════════════════════")

    prefix = "compress"
    if args.quantize:
        prefix = f"compress_{args.bits}bit"
    path = save_results_json(report, default_dir="checkpoints/compress",
                             prefix=prefix)
    print(f"\nResults: {path}")


def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Run REVO pipeline phases (I-V + X-XI)")
    _add_common_args(p)
    p.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> None:
    from revo.pipeline import PhaseRunner
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_tokenizer(args.model)
    texts = gen_texts(args.prompts)
    runner = PhaseRunner({"phase1": {"seed": args.seed}})
    report = runner.run_all(model, tokenizer, texts, max_len=args.max_length)

    print(f"\n{'─'*60}")
    print(f"{'Phase':<20} {'Status':<12} {'NLL Δ':<14} {'Time ratio':<12} {'Params Δ':<12}")
    print(f"{'─'*60}")
    for key in ["phase1", "phase2", "phase3", "phase4", "phase5", "phase_ephemeral", "phase10", "phase11"]:
        p = report.get(key, {})
        nd = f"{p.get('nll_delta', 0):+.6e}" if p.get("nll_delta") is not None else "—"
        tr = f"{p.get('time_ratio', 1):.4f}×" if p.get("time_ratio") is not None else "—"
        pd = f"{p.get('params_delta', 0):+d}" if p.get("params_delta") is not None else "—"
        print(f"{key:<20} {p.get('status','?'):<12} {nd:<14} {tr:<12} {pd:<12}")
    print(f"{'─'*60}")
    print(f"Baseline NLL: {report.get('baseline', {}).get('nll', '—'):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO — inference-time transformations for LLMs")
    subparsers = ap.add_subparsers(dest="command", required=True)
    register_compress_subcommand(subparsers)
    register_run_subcommand(subparsers)

    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(1)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
