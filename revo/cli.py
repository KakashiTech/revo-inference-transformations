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
    for key in ["phase1", "phase2", "phase3", "phase4", "phase5", "phase_ephemeral", "phase10", "phase11", "phase12", "phase12b", "phase12c", "phase12d"]:
        p = report.get(key, {})
        nd = f"{p.get('nll_delta', 0):+.6e}" if p.get("nll_delta") is not None else "—"
        tr = f"{p.get('time_ratio', 1):.4f}×" if p.get("time_ratio") is not None else "—"
        pd = f"{p.get('params_delta', 0):+d}" if p.get("params_delta") is not None else "—"
        print(f"{key:<20} {p.get('status','?'):<12} {nd:<14} {tr:<12} {pd:<12}")
    print(f"{'─'*60}")
    print(f"Baseline NLL: {report.get('baseline', {}).get('nll', '—'):.6f}")


def register_shard_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("shard", help="Shard model weights to disk (per-layer)")
    _add_common_args(p)
    p.add_argument("--output", default="./shards", help="Output directory for shards")
    p.set_defaults(func=cmd_shard)


def cmd_shard(args: argparse.Namespace) -> None:
    from revo.streaming import shard_model
    model, tokenizer = load_model_tokenizer(args.model)  # noqa: F841
    print(f"Sharding {args.model} → {args.output}")
    info = shard_model(model, args.output)
    total_mb = sum(info.values()) / 1024 / 1024
    for k, v in info.items():
        print(f"  {k}: {v/1024:.1f} KB")
    print(f"Total: {total_mb:.1f} MB")
    print(f"To run: revo stream --model {args.model} --shards {args.output}")


def register_stream_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("stream", help="Run inference with streaming weights")
    _add_common_args(p)
    p.add_argument("--shards", default="./shards", help="Shard directory")
    p.add_argument("--prompt", default="The meaning of life is",
                   help="Input prompt for generation")
    p.add_argument("--max-new", type=int, default=20,
                   help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--compare", action="store_true",
                   help="Compare full vs streaming memory")
    p.set_defaults(func=cmd_stream)


def cmd_stream(args: argparse.Namespace) -> None:
    from revo.streaming import stream_generate, compare_memory, shard_model
    import os

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_tokenizer(args.model)

    if not os.path.isdir(args.shards):
        print(f"Sharding model to {args.shards}...")
        shard_model(model, args.shards)

    if args.compare:
        print("Comparing full load vs streaming...")
        report = compare_memory(model, tokenizer, args.shards,
                                gen_texts(args.prompts), max_length=args.max_length)
        print(f"  Weight savings: {report['theory']['savings_pct']:.1f}%")
        print(f"  NLL delta:      {report['streaming']['nll_delta']:+.6f}")
        print(f"  Time ratio:     {report['streaming']['time_ratio']:.1f}×")
    else:
        print(f"Generating with streaming weights...")
        text, meta = stream_generate(
            model, tokenizer, args.prompt, args.shards,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {text}")
        print(f"Time:   {meta['total_time_s']:.2f}s ({meta['mean_time_per_token_s']*1000:.0f}ms/tok)")


def register_law_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("law", help="Generative WeightLaw operations")
    _add_common_args(p)
    p.add_argument("--action",
                   choices=["build", "stream", "benchmark", "finetune",
                            "generate", "train-e2e", "field-only"],
                   default="build", help="Law action")
    p.add_argument("--rank", type=int, default=16, help="SVD rank for law")
    p.add_argument("--small-dim", type=int, default=8, help="Bottleneck dimension")
    p.add_argument("--hidden-dim", type=int, default=256, help="Law MLP hidden dim")
    p.add_argument("--prompt", default="The meaning of life is",
                   help="Input prompt for generation")
    p.add_argument("--max-new", type=int, default=20,
                   help="Max new tokens to generate")
    p.add_argument("--steps", type=int, default=500,
                   help="Training steps for e2e or field training")
    p.add_argument("--data-samples", type=int, default=2000,
                   help="Number of wikitext samples for training")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Training learning rate")
    p.add_argument("--kl-lambda", type=float, default=0.0,
                   help="KL divergence weight (0=off)")
    p.add_argument("--lambda-sparse", type=float, default=0.3,
                   help="Sparsity weight for field training")
    p.add_argument("--cognitive", action="store_true",
                   help="Enable cognitive field (Phase XII-c)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving checkpoint")
    p.set_defaults(func=cmd_law)


def cmd_law(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_tokenizer(args.model)

    from revo.law_streaming import (
        build_law, pretrain_law, finetune_law, law_stream_forward,
        law_generate,
        _extract_svd_targets, law_info, law_compare_memory, _save_block_templates,
    )
    from revo.streaming import _free_all_block_weights, _d_model

    if args.action == "build":
        print(f"Building WeightLaw for {args.model}...")
        law = build_law(model, rank=args.rank, small_dim=args.small_dim,
                        hidden_dim=args.hidden_dim,
                        cognitive=args.cognitive, cognitive_field=args.cognitive)
        if args.cognitive:
            print(f"  Cognitive field: enabled")
        targets = _extract_svd_targets(model, rank=args.rank)
        pretrain_law(law, targets, steps=100, lr=1e-3, verbose=True)
        info = law_info(law)
        print(f"\nLaw built: {info['n_params']:,} params ({info['size_mb']:.1f} MB)")
        print(f"  Rank: {info['rank']}, Heads: {info['head_names']}")
        if hasattr(law, 'field') and law.field is not None:
            fs = law.field(torch.randn(1, 1, _d_model(model)))
            print(f"  Field: {fs}")

    elif args.action in ("stream", "generate"):
        law = build_law(model, rank=args.rank, small_dim=args.small_dim,
                        hidden_dim=args.hidden_dim,
                        cognitive=args.cognitive, cognitive_field=args.cognitive)
        # Delta mode (rank<=8) doesn't need pretraining — zero-shot quality
        if args.rank > 8:
            targets = _extract_svd_targets(model, rank=args.rank)
            pretrain_law(law, targets, steps=100, lr=1e-3, verbose=False)
        print(f"Generating with law-streamed weights (KV cache)...")
        if args.cognitive:
            print(f"  Cognitive field active")
        templates = _save_block_templates(model)
        _free_all_block_weights(model)
        text, meta = law_generate(
            model, tokenizer, args.prompt, law,
            max_new_tokens=args.max_new,
            temperature=1.0, top_k=None, top_p=None,
            block_templates=templates, verbose=True,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {text}")
        print(f"\nTiming: {meta['mean_time_per_token_s']*1000:.0f}ms/tok "
              f"({meta['n_new_tokens']} tokens)")

    elif args.action == "benchmark":
        texts = gen_texts(args.prompts)
        law = build_law(model, rank=args.rank, small_dim=args.small_dim,
                        hidden_dim=args.hidden_dim)
        targets = _extract_svd_targets(model, rank=args.rank)
        pretrain_law(law, targets, steps=100, lr=1e-3, verbose=False)
        print("Benchmarking law vs full model...")
        report = law_compare_memory(model, tokenizer, law, texts,
                                    max_length=args.max_length, free_blocks=True)
        f, ls, l = report["full"], report["law_streaming"], report["law"]
        print(f"  Full: NLL={f['nll']:.4f}  RSS={f['rss_mb']:.0f}MB  Time={f['time_s']:.2f}s")
        print(f"  Law:  NLL={ls['nll']:.4f} (Δ={ls['nll_delta']:+.2f})  "
              f"Peak RSS={ls['rss_peak_mb']:.0f}MB  "
              f"Time={ls['time_s']:.2f}s ({ls['time_ratio']:.1f}x)")
        print(f"  Law size: {l['n_params']:,} params ({l['n_params_mb']:.0f} MB)")

    elif args.action == "finetune":
        texts = gen_texts(args.prompts)
        law = build_law(model, rank=args.rank, small_dim=args.small_dim,
                        hidden_dim=args.hidden_dim)
        targets = _extract_svd_targets(model, rank=args.rank)
        pretrain_law(law, targets, steps=100, lr=1e-3, verbose=False)
        print(f"Fine-tuning law on {len(texts)} texts...")
        losses = finetune_law(law, model, tokenizer, texts, steps=10,
                              lr=5e-5, max_length=args.max_length, verbose=True)
        print(f"  NLL: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[-1]-losses[0]:+.4f})")

    elif args.action == "train-e2e":
        """End-to-end NLL training on wikitext."""
        from train_law_e2e import train_e2e as train_fn
        print(f"End-to-end law training: rank={args.rank}, steps={args.steps}")
        print(f"  lr={args.lr}, kl={args.kl_lambda}, batch={args.batch_size}")
        train_fn(
            model_name=args.model,
            rank=args.rank,
            small_dim=args.small_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            steps=args.steps,
            batch_size=args.batch_size,
            max_samples=args.data_samples,
            lambda_kl=args.kl_lambda,
        )

    elif args.action == "field-only":
        """Train only the CognitiveField (no law training)."""
        from train_field_router import train_field_router as train_fn
        print(f"Field-only training: rank={args.rank}, steps={args.steps}")
        print(f"  λ_sparse={args.lambda_sparse}, lr={args.lr}")
        train_fn(
            model_name=args.model,
            lr=args.lr,
            lambda_sparse=args.lambda_sparse,
            steps=args.steps,
            batch_size=args.batch_size,
            rank=args.rank,
            hidden_dim=args.hidden_dim,
            cognitive=args.cognitive,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO — inference-time transformations for LLMs")
    subparsers = ap.add_subparsers(dest="command", required=True)
    register_compress_subcommand(subparsers)
    register_run_subcommand(subparsers)
    register_shard_subcommand(subparsers)
    register_stream_subcommand(subparsers)
    register_law_subcommand(subparsers)

    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(1)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
