"""Integration test for FDMFilterBank (I.4 Tesla Resonance Routing)."""

import torch

from revo.archive.fdm_rtd import FDMFilterBank


def test_fdm_forward_shape():
    """Verify that FDMFilterBank preserves feature dimensions."""
    n_bands = 4
    in_features = 32
    out_features = 32
    batch = 2

    fdm = FDMFilterBank(n_bands, in_features, out_features)
    x = torch.randn(batch, in_features)
    y = fdm(x)

    assert y.shape == (batch, out_features), \
        f"Expected ({batch}, {out_features}), got {y.shape}"
    print(f"FDMFilterBank forward OK – input {x.shape} → output {y.shape}")


def test_fdm_multi_dim():
    """FDMFilterBank accepts arbitrary leading dims."""
    fdm = FDMFilterBank(2, 16, 16)
    x = torch.randn(3, 5, 16)
    y = fdm(x)
    assert y.shape == (3, 5, 16), f"Got {y.shape}"
    print(f"Multi-dim forward OK – {x.shape} → {y.shape}")


def test_fdm_band_independence():
    """Check that bands actually process independently (weights differ)."""
    fdm = FDMFilterBank(4, 32, 32)
    x = torch.randn(1, 32)
    # Run twice
    y1 = fdm(x)
    # Change one band's linear weight
    with torch.no_grad():
        old = fdm.band_linears[0].weight.clone()
        fdm.band_linears[0].weight.add_(torch.randn_like(old) * 10)
    y2 = fdm(x)
    assert not torch.allclose(y1, y2, atol=1e-4), "Bands should be independent"
    print("Band independence check OK")


if __name__ == "__main__":
    test_fdm_forward_shape()
    test_fdm_multi_dim()
    test_fdm_band_independence()
    print("\nAll FDMFilterBank tests passed.")
