"""Unit tests for the parametric insurance payout engine."""
import pytest
import torch

from src.analytics.payout import compute_payout, compute_expected_loss_ratio


class TestComputePayout:
    def test_below_attachment_zero_payout(self):
        """SVaR below attachment → zero payout."""
        svar = torch.tensor([10.0, 0.0, 19.9])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert torch.all(payout == 0.0), f"expected all zeros, got {payout}"

    def test_above_cap_full_payout(self):
        """SVaR at or above cap → full coverage payout."""
        svar = torch.tensor([60.0, 100.0, 999.0])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert torch.allclose(payout, torch.full((3,), 1_000_000.0)), f"expected 1M, got {payout}"

    def test_midpoint_half_payout(self):
        """SVaR at midpoint between attachment and cap → half coverage."""
        svar = torch.tensor([40.0])  # midpoint of [20, 60]
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.item() == pytest.approx(500_000.0, abs=1.0)

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        svar = torch.zeros(3, 5)  # batch=3, member=5
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.shape == (3, 5)

    def test_attachment_equals_cap_raises(self):
        """attachment == cap is undefined (division by zero)."""
        with pytest.raises(ValueError, match="cap.*must be.*greater than.*attachment"):
            compute_payout(torch.tensor([30.0]), attachment=20.0, cap=20.0, coverage=1e6)

    def test_cap_less_than_attachment_raises(self):
        """cap < attachment is nonsensical."""
        with pytest.raises(ValueError, match="cap.*must be.*greater than.*attachment"):
            compute_payout(torch.tensor([30.0]), attachment=60.0, cap=20.0, coverage=1e6)

    def test_exact_attachment_boundary(self):
        """SVaR exactly at attachment → zero payout (not partial)."""
        svar = torch.tensor([20.0])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.item() == pytest.approx(0.0, abs=1e-5)


class TestComputeExpectedLossRatio:
    def test_full_loss_ratio_at_cap(self):
        """All members at cap SDD → loss_ratio_95 = 1.0."""
        sdd = torch.full((2, 8), 60.0)  # all members at cap
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert torch.allclose(result["loss_ratio_95"], torch.ones(2), atol=1e-4)

    def test_zero_loss_ratio_below_attachment(self):
        """All members below attachment → loss_ratio_95 = 0.0."""
        sdd = torch.full((2, 8), 5.0)
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert torch.all(result["loss_ratio_95"] == 0.0)

    def test_output_keys_present(self):
        """Result dict contains the four required keys."""
        sdd = torch.rand(2, 8) * 80.0
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        for key in ("svar_95", "payout_95", "loss_ratio_95", "payout_50"):
            assert key in result, f"missing key: {key}"

    def test_loss_ratio_in_unit_interval(self):
        """Loss ratio is always in [0, 1]."""
        sdd = torch.rand(10, 8) * 100.0  # random SDDs spanning [0, 100]
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert (result["loss_ratio_95"] >= 0.0).all()
        assert (result["loss_ratio_95"] <= 1.0).all()
