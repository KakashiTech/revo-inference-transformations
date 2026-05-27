"""Shared fixtures for REVO tests."""
from __future__ import annotations

import pytest
import torch
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def device():
    return torch.device("cpu")
