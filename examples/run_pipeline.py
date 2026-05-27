"""Minimal test harness for PhaseRunner with a mock model."""
from __future__ import annotations

import torch
import torch.nn as nn

from revo.pipeline import PhaseRunner


class TinyLinearModel(nn.Module):
    """Minimal causal LM stub for pipeline testing."""

    def __init__(self, vocab_size: int = 256, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        h = self.linear(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type("LMOutput", (), {"loss": loss, "logits": logits, "past_key_values": None})()


class TinyTokenizer:
    """Minimal tokenizer stub that returns integer token sequences."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token = 0
        self.pad_token = 0
        self.bos_token_id = 0

    class _Encoding:
        def __init__(self, t):
            self.input_ids = t
            self.attention_mask = torch.ones_like(t)
        def items(self):
            return [("input_ids", self.input_ids), ("attention_mask", self.attention_mask)]
        def __getitem__(self, k):
            return getattr(self, k)

    def __call__(self, text: str, return_tensors=None, truncation=None, max_length=None):
        tokens = [hash(c) % self.vocab_size for c in text[:max_length or len(text)]]
        if max_length and len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))
        t = torch.tensor([tokens[:max_length or len(tokens)]])
        return self._Encoding(t)


def main() -> None:
    vocab_size = 256
    model = TinyLinearModel(vocab_size=vocab_size)
    tokenizer = TinyTokenizer(vocab_size=vocab_size)
    texts = ["hello world", "test prompt", "REVO pipeline"]

    runner = PhaseRunner()
    report = runner.run_all(model, tokenizer, texts, max_len=16)

    # Verify structure
    required_keys = ["baseline", "phase1", "phase2", "phase3", "phase4", "phase5", "summary"]
    for k in required_keys:
        assert k in report, f"Missing key: {k}"
        assert isinstance(report[k], dict), f"{k} should be a dict"

    assert "status" in report["phase1"], "phase1 missing status"
    assert "summary" in report
    assert report["summary"]["total_phases"] == 5

    print("=" * 60)
    print("Pipeline report structure: OK")
    print("=" * 60)

    for key in required_keys:
        val = report[key]
        if isinstance(val, dict) and "status" in val:
            print(f"  {key}: status={val['status']}")
        else:
            print(f"  {key}: {list(val.keys()) if isinstance(val, dict) else val}")

    print(f"\nSummary: {report['summary']}")
    print("\nAll assertions passed. Pipeline runner works correctly.")


if __name__ == "__main__":
    main()
