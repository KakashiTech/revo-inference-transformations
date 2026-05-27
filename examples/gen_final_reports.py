from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


VOCAB_SIZE = 64
HIDDEN_DIM = 32
NUM_LAYERS = 4
SEQ_LEN = 10


class MockEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class MockOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: Tuple[torch.Tensor, ...], loss: torch.Tensor):
        self.logits = logits
        self.hidden_states = hidden_states
        self.loss = loss


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.hidden_dim = HIDDEN_DIM
        for i in range(3):
            sub = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            setattr(self, f"mock_linear_{i}", sub)

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=False, use_cache=False):
        B, T = input_ids.shape
        logits = torch.randn(B, T, self.vocab_size)
        hs = tuple(torch.randn(B, T, self.hidden_dim) for _ in range(NUM_LAYERS + 1))
        loss_val = torch.tensor(random.uniform(4.0, 12.0), requires_grad=True)
        return MockOutput(logits, hs, loss_val)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_output_embeddings(self):
        return None

    def get_input_embeddings(self):
        return None


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.eos_token = "<|endoftext|>"
        self.pad_token = None
        self.vocab_size = VOCAB_SIZE

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        B = 1
        T = min(SEQ_LEN, max_length or SEQ_LEN)
        return MockEncoding(
            input_ids=torch.randint(0, self.vocab_size, (B, T)),
            attention_mask=torch.ones(B, T, dtype=torch.long),
        )


def _mock_auto(model_name: str) -> MockModel:
    return MockModel()


def _mock_tokenizer(model_name: str) -> MockTokenizer:
    return MockTokenizer()


def run_phase_vi(out_path: str) -> Dict:
    import revo.regimes
    revo.regimes._device = lambda: torch.device("cpu")

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_mock_auto),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_mock_tokenizer),
    ):
        report = revo.regimes.evaluate_regimes(
            model_name="mock-model",
            seeds=[0],
            prompts=5,
            max_len=SEQ_LEN,
            config=revo.regimes.RegimeConfig(topk_entropy=16),
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_phase_vii(out_path: str) -> Dict:
    import revo.probcal
    revo.probcal._device = lambda: torch.device("cpu")

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_mock_auto),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_mock_tokenizer),
    ):
        report = revo.probcal.evaluate_probcal(
            model_name="mock-model",
            seeds=[0],
            prompts=5,
            max_len=SEQ_LEN,
            config=revo.probcal.ProbCalConfig(steps=2, topk_eval=16),
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_phase_viii(out_path: str) -> Dict:
    import revo.implicit
    revo.implicit._device = lambda: torch.device("cpu")

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_mock_auto),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_mock_tokenizer),
    ):
        report = revo.implicit.evaluate_implicit(
            model_name="mock-model",
            seeds=[0],
            prompts=5,
            max_len=SEQ_LEN,
            config=revo.implicit.ImplicitConfig(codebook_k=4, iters=2, q_quantile=0.5),
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_phase_ix(out_path: str) -> Dict:
    import revo.biocomp
    revo.biocomp._device = lambda: torch.device("cpu")

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_mock_auto),
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=_mock_tokenizer),
    ):
        report = revo.biocomp.evaluate_biocomp(
            model_name="mock-model",
            seeds=[0],
            prompts=5,
            max_len=SEQ_LEN,
            config=revo.biocomp.BioCompConfig(
                act_quantile=0.75, topk_eval=16, energy_warmup=0, energy_runs=1
            ),
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    base_dir = os.path.join("quality", "final_runs")
    os.makedirs(base_dir, exist_ok=True)

    phases = [
        ("VI", "regimes", run_phase_vi, os.path.join(base_dir, "regimes_report.json")),
        ("VII", "probcal", run_phase_vii, os.path.join(base_dir, "probcal_report.json")),
        ("VIII", "implicit", run_phase_viii, os.path.join(base_dir, "implicit_report.json")),
        ("IX", "biocomp", run_phase_ix, os.path.join(base_dir, "biocomp_report.json")),
    ]

    summaries = []
    for phase_num, phase_name, runner, out_path in phases:
        print(f"--- Phase {phase_num} ({phase_name}) ---")
        t0 = time.perf_counter()
        report = runner(out_path)
        elapsed = time.perf_counter() - t0
        summary = report.get("summary", {})
        print(f"  Summary keys: {list(summary.keys())}")
        for k, v in summary.items():
            print(f"    {k}: {v}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Saved: {out_path}\n")
        summaries.append({"phase": phase_num, "name": phase_name, "summary": summary, "path": out_path})

    print("=" * 60)
    print("ALL REPORTS GENERATED")
    print("=" * 60)
    for s in summaries:
        print(f"  Phase {s['phase']} ({s['name']}): {s['path']}")
        print(f"    Summary: {json.dumps(s['summary'])}")

    print("\n--- Verification ---")
    all_ok = True
    for s in summaries:
        exists = os.path.isfile(s["path"])
        size_kb = os.path.getsize(s["path"]) / 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {s['path']}: {status} ({size_kb:.1f} KB)")
        if not exists:
            all_ok = False

    if all_ok:
        print("\nAll 4 JSON files created successfully.")
    else:
        print("\nSome files are missing!")

    return summaries


if __name__ == "__main__":
    main()
