# tests/test_svar.py
"""Unit tests for Stochastic Value-at-Risk estimation."""
import pytest
import torch

from src.analytics.svar import compute_svar, compute_ensemble_stats


class TestComputeSvar:
    def test_output_shape_is_batch(self, sdd_tensor):
        """SVaR output has shape (batch,) — one value per location/season."""
        svar = compute_svar(sdd_tensor, quantile=0.95)
        assert svar.shape == (2,)

    def test_svar_95_captures_high_member(self, sdd_tensor):
        """
        sdd_tensor batch 0: 7 members near 1-2 °C·day, member 7 = 10 °C·day.
        SVaR_95 of 8 members: top 5% → member 7 (10.0) dominates.
        """
        svar = compute_svar(sdd_tensor, quantile=0.95)
        # torch.quantile([1.0, 1.5, 2.0, 1.2, 1.8, 1.3, 1.6, 10.0], 0.95)
        # linear interp at index 0.95*(8-1)=6.65 between sorted[6]=2.0 and sorted[7]=10.0:
        # 2.0 + 0.65*8.0 = 7.20
        assert abs(svar[0].item() - 7.20) < 0.01, \
            f"SVaR_95 expected ~7.20 for this fixture; got {svar[0].item():.3f}"

    def test_svar_50_is_median(self):
        """SVaR at quantile=0.50 is the ensemble median."""
        sdd = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # (1, 8)
        svar = compute_svar(sdd, quantile=0.50)
        assert abs(svar[0].item() - 4.5) < 0.01

    def test_uniform_ensemble_all_quantiles_equal(self):
        """All members identical → all quantiles equal."""
        sdd = torch.full((3, 64), 7.5)
        for q in [0.05, 0.50, 0.95, 0.99]:
            svar = compute_svar(sdd, quantile=q)
            torch.testing.assert_close(svar, torch.full((3,), 7.5))

    def test_quantile_monotone(self, sdd_tensor):
        """SVaR_99 >= SVaR_95 >= SVaR_50 for the same batch."""
        s50 = compute_svar(sdd_tensor, quantile=0.50)
        s95 = compute_svar(sdd_tensor, quantile=0.95)
        s99 = compute_svar(sdd_tensor, quantile=0.99)
        assert (s99 >= s95).all()
        assert (s95 >= s50).all()

    def test_invalid_quantile_raises(self, sdd_tensor):
        """Quantile outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="quantile"):
            compute_svar(sdd_tensor, quantile=1.5)
        with pytest.raises(ValueError, match="quantile"):
            compute_svar(sdd_tensor, quantile=0.0)

    def test_nonnegative_sdd_gives_nonnegative_svar(self):
        """All-zero SDD ensemble → SVaR = 0."""
        sdd = torch.zeros(4, 64)
        svar = compute_svar(sdd, quantile=0.95)
        assert (svar >= 0).all()
        torch.testing.assert_close(svar, torch.zeros(4))


class TestComputeEnsembleStats:
    def test_output_keys(self, sdd_tensor):
        """Returns dict with expected keys."""
        stats = compute_ensemble_stats(sdd_tensor)
        for key in ("mean", "std", "svar_50", "svar_90", "svar_95", "svar_99"):
            assert key in stats, f"Missing key: {key}"

    def test_shapes(self, sdd_tensor):
        """All stat tensors have shape (batch,)."""
        stats = compute_ensemble_stats(sdd_tensor)
        for key, val in stats.items():
            assert val.shape == (2,), f"{key} shape mismatch: {val.shape}"

    def test_mean_is_correct(self):
        """Mean computed correctly for a known input."""
        sdd = torch.tensor([[2.0, 4.0, 6.0, 8.0]])  # mean = 5.0
        stats = compute_ensemble_stats(sdd)
        torch.testing.assert_close(stats["mean"], torch.tensor([5.0]))

    def test_std_uses_population_correction(self):
        """std uses N (population) not N-1 (sample) denominator."""
        # [1, 3] — population std = 1.0, sample std = sqrt(2) ≈ 1.414
        sdd = torch.tensor([[1.0, 3.0]])
        stats = compute_ensemble_stats(sdd)
        torch.testing.assert_close(stats["std"], torch.tensor([1.0]))
