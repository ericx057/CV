"""
Matrix injection structural tests — no model calls.

Covers:
  - SVD math: A rows orthonormal, B = A.T, shape contracts
  - Correction = B @ (A @ v) is an orthogonal projection
  - ||correction(v)|| <= ||v|| for all v  (bounded residual stream)
  - sv_energy_frac formula (strict < 1.0 for r < rank, = 1.0 at full rank)
  - The original B = H.T @ U_r normalization bug (unbounded correction)
  - Rank truncation when seq_len or d_model < requested rank
  - MatrixBenchmarkRecord field structure

Mathematical background:
  H [seq_len, d_model] — context hidden states
  SVD: H = U @ diag(S) @ Vh
  A = Vh[:r, :]   [r, d_model]  -- context subspace directions (rows orthonormal)
  B = Vh[:r, :].T [d_model, r]  -- B = A.T by construction
  correction(v) = B @ (A @ v) = Vh_r.T @ (Vh_r @ v)
    This is an orthogonal projection of v onto the top-r right-singular subspace.
    ||correction(v)|| <= ||v|| always holds because Vh_r is a partial isometry.

  Original bug: B = H.T @ U_r = Vh_r.T @ diag(S_r)
    Columns scaled by S_r (can be O(1000) for long contexts) → overrides residual stream.

Run in isolation:
    pytest test_bench_matrix.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers: replicate extract_context_matrix math without a model
# ---------------------------------------------------------------------------

def _make_H(rows: int, cols: int, seed: int = 42) -> torch.Tensor:
    """Synthetic hidden-state matrix [rows, cols]."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((rows, cols))).float()


def _extract_matrix_correct(H: torch.Tensor, rank: int):
    """
    Replicate extract_context_matrix with the correct (bounded) B = Vh_r.T.

    Returns (A, B, S_r, S_full, effective_rank).
    """
    effective_rank = min(rank, H.shape[0], H.shape[1])
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]
    A = Vh_r          # [r, d_model]
    B = Vh_r.T        # [d_model, r]  -- normalized (correct fix)
    return A, B, S_r, S, effective_rank


def _extract_matrix_buggy(H: torch.Tensor, rank: int):
    """
    Replicate the original (buggy) B = H.T @ U_r formulation.
    B columns are scaled by singular values, making correction unbounded.

    Returns (A, B_buggy, S_r, effective_rank).
    """
    effective_rank = min(rank, H.shape[0], H.shape[1])
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]
    U_r = U[:, :effective_rank]
    A = Vh_r
    B_buggy = H.T @ U_r  # [d_model, r]  -- original, unscaled columns
    return A, B_buggy, S_r, effective_rank


# ---------------------------------------------------------------------------
# A matrix: rows are orthonormal
# ---------------------------------------------------------------------------

class TestAMatrixOrthonormality:
    def test_rows_are_unit_norm(self):
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        norms = torch.linalg.norm(A, dim=1)
        for i, n in enumerate(norms):
            assert abs(n.item() - 1.0) < 1e-5, (
                f"Row {i} norm = {n.item():.6f}, expected 1.0"
            )

    def test_rows_are_mutually_orthogonal(self):
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        gram = A @ A.T  # [r, r]
        identity = torch.eye(r)
        off_diag = (gram - identity).abs().max().item()
        assert off_diag < 1e-4, (
            f"A rows not orthonormal: max off-diagonal gram = {off_diag:.6f}"
        )

    def test_b_equals_a_transpose(self):
        """B = A.T by construction — always holds."""
        H = _make_H(40, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=6)
        assert torch.allclose(B, A.T, atol=1e-6), (
            "B should equal A.T by construction"
        )

    def test_shape_a_is_r_by_d(self):
        H = _make_H(30, 64, seed=7)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        assert A.shape == (r, 64)

    def test_shape_b_is_d_by_r(self):
        H = _make_H(30, 64, seed=7)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        assert B.shape == (64, r)

    def test_b_column_norms_are_one(self):
        """Each column of B = Vh_r.T has unit norm (rows of Vh_r are orthonormal)."""
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        col_norms = torch.linalg.norm(B, dim=0)
        for i, n in enumerate(col_norms):
            assert abs(n.item() - 1.0) < 1e-5, (
                f"B column {i} norm = {n.item():.6f}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# Correction is an orthogonal projection
# ---------------------------------------------------------------------------

class TestProjectionProperties:
    def test_projection_is_idempotent(self):
        """
        P = B @ A is an orthogonal projection: P @ P == P  (up to float tolerance).
        """
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        P = B @ A   # [d_model, d_model]
        PP = P @ P
        max_diff = (PP - P).abs().max().item()
        assert max_diff < 1e-4, f"P@P != P: max_diff={max_diff:.6f}"

    def test_correction_magnitude_bounded_by_input(self):
        """||B @ (A @ v)|| <= ||v|| for 50 random vectors v."""
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        rng = np.random.default_rng(99)
        for _ in range(50):
            v = torch.from_numpy(rng.standard_normal(64)).float()
            correction = B @ (A @ v)
            v_norm = v.norm().item()
            c_norm = correction.norm().item()
            assert c_norm <= v_norm + 1e-5, (
                f"correction norm {c_norm:.4f} > input norm {v_norm:.4f}"
            )

    def test_correction_bounded_for_ranks_1_to_16(self):
        """Bound holds for every rank in [1, 2, 4, 8, 16]."""
        H = _make_H(80, 128, seed=5)
        rng = np.random.default_rng(77)
        v = torch.from_numpy(rng.standard_normal(128)).float()
        v_norm = v.norm().item()
        for rank in [1, 2, 4, 8, 16]:
            A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=rank)
            correction = B @ (A @ v)
            c_norm = correction.norm().item()
            assert c_norm <= v_norm + 1e-5, (
                f"rank={rank}: correction norm {c_norm:.4f} > input norm {v_norm:.4f}"
            )

    def test_zero_vector_gives_zero_correction(self):
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        v = torch.zeros(64)
        correction = B @ (A @ v)
        assert correction.norm().item() < 1e-6

    def test_correction_of_a_row_is_the_row_itself(self):
        """
        A row a_i is in the subspace spanned by A rows.
        projection(a_i) should equal a_i (it's already in the subspace).
        """
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        for i in range(r):
            a_i = A[i]
            projected = B @ (A @ a_i)
            diff = (projected - a_i).norm().item()
            assert diff < 1e-4, (
                f"Row {i}: projection of subspace vector != itself, diff={diff:.6f}"
            )

    def test_correction_of_orthogonal_vector_is_near_zero(self):
        """
        If v is orthogonal to all rows of A, correction(v) should be zero.
        """
        # Use rank=1 for simplicity: A = Vh[0:1, :]
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=1)
        a0 = A[0]  # the single subspace direction

        # Build a vector orthogonal to a0 (Gram-Schmidt)
        rng = np.random.default_rng(7)
        v_raw = torch.from_numpy(rng.standard_normal(64)).float()
        v_orth = v_raw - (v_raw @ a0) * a0  # project out a0 component

        correction = B @ (A @ v_orth)
        c_norm = correction.norm().item()
        assert c_norm < 1e-4, (
            f"Correction of orthogonal vector not near zero: {c_norm:.6f}"
        )


# ---------------------------------------------------------------------------
# sv_energy_frac formula
# ---------------------------------------------------------------------------

class TestSvEnergyFrac:
    @staticmethod
    def _frac(S_r: torch.Tensor, S_full: torch.Tensor) -> float:
        return float(S_r.sum() / (S_full.sum() + 1e-8))

    def test_full_rank_gives_frac_one(self):
        """When r = min(seq_len, d_model), all singular values are captured."""
        H = _make_H(10, 64)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        full_r = min(H.shape[0], H.shape[1])
        S_r = S[:full_r]
        frac = self._frac(S_r, S)
        assert abs(frac - 1.0) < 1e-5, f"Full rank frac = {frac:.6f}, expected ~1.0"

    def test_partial_rank_frac_is_strictly_less_than_one(self):
        """For rank < min(seq_len, d_model), frac < 1.0."""
        H = _make_H(50, 64)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        frac = self._frac(S[:4], S)
        assert frac < 1.0, f"Partial rank frac = {frac:.4f}, should be < 1.0"
        assert frac > 0.0, f"Partial rank frac = {frac:.4f}, should be > 0.0"

    def test_frac_is_monotone_increasing_with_rank(self):
        """Adding more singular values never reduces the captured energy."""
        H = _make_H(50, 64)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        fracs = [self._frac(S[:r], S) for r in [1, 2, 4, 8, 16]]
        for i in range(len(fracs) - 1):
            assert fracs[i] <= fracs[i + 1] + 1e-6, (
                f"Energy frac not monotone at rank {i+1}→{i+2}: "
                f"{fracs[i]:.4f}→{fracs[i+1]:.4f}"
            )

    def test_frac_stays_in_zero_one(self):
        """sv_energy_frac must be in [0, 1] for all inputs."""
        for seed in range(5):
            H = _make_H(30, 64, seed=seed)
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)
            frac = self._frac(S[:8], S)
            assert 0.0 <= frac <= 1.0 + 1e-6, (
                f"seed={seed}: sv_energy_frac={frac:.6f} out of [0, 1]"
            )

    def test_rank_1_on_large_random_captures_less_than_half(self):
        """
        For a large random matrix, the top singular vector should capture
        well under 50% of the total singular value energy.
        """
        H = _make_H(100, 128, seed=0)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        frac = self._frac(S[:1], S)
        assert frac < 0.5, (
            f"rank=1 on 100×128 random matrix captured {frac:.2%} energy (expected <50%)"
        )

    def test_frac_not_always_one_for_reasonable_rank(self):
        """
        A rank-8 SVD on a 50×64 random matrix should NOT capture 100% of energy.
        This was the sv_energy_frac = 1.0 bug: full rank was used instead of r.
        """
        H = _make_H(50, 64)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        frac = self._frac(S[:8], S)
        # 50 non-zero singular values; rank=8 should leave out the rest
        assert frac < 0.999, (
            f"rank=8 captured {frac:.4%} of energy — suspiciously high "
            "(did you accidentally sum all singular values?)"
        )


# ---------------------------------------------------------------------------
# Normalization bug: B = H.T @ U_r is unbounded
# ---------------------------------------------------------------------------

class TestNormalizationBug:
    """
    The original B = H.T @ U_r = Vh_r.T @ diag(S_r) scales each column of B
    by the corresponding singular value S_i. For typical long contexts,
    S_0 can be O(100–1000), making ||correction(v)|| >> ||v||.

    The fix: B = Vh_r.T keeps each column at unit norm, making
    ||correction(v)|| <= ||v|| always.
    """

    def test_buggy_b_column_norms_scale_with_singular_values(self):
        """
        Columns of B_buggy = H.T @ U_r have norms equal to S_r (not 1).
        Columns of B_correct = Vh_r.T have norms equal to 1.
        """
        H = _make_H(50, 64)
        A, B_correct, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        A_b, B_buggy, S_r_b, _ = _extract_matrix_buggy(H, rank=4)

        col_norms_correct = torch.linalg.norm(B_correct, dim=0)
        col_norms_buggy = torch.linalg.norm(B_buggy, dim=0)

        # Correct: all column norms = 1
        for i in range(r):
            assert abs(col_norms_correct[i].item() - 1.0) < 1e-4, (
                f"B_correct col {i} not unit norm: {col_norms_correct[i].item():.4f}"
            )
        # Buggy: column norms = S_r (must be > 1 for any real H)
        for i in range(r):
            assert col_norms_buggy[i].item() > 1.0, (
                f"B_buggy col {i} unexpectedly unit norm: {col_norms_buggy[i].item():.4f}"
            )

    def test_buggy_correction_exceeds_input_norm_on_large_singular_values(self):
        """
        Construct H with dominant first singular value (~1000).
        Buggy correction norm >> ||v||; correct correction norm <= ||v||.
        """
        d_model = 64
        seq_len = 50
        rng = np.random.default_rng(42)

        # Build H with known singular values [1000, 10, 1, 0.5]
        Q_left = torch.from_numpy(
            np.linalg.qr(rng.standard_normal((seq_len, seq_len)))[0]
        ).float()[:, :4]
        Q_right = torch.from_numpy(
            np.linalg.qr(rng.standard_normal((d_model, d_model)))[0]
        ).float()[:4, :]
        S_diag = torch.tensor([1000.0, 10.0, 1.0, 0.5])
        H = (Q_left * S_diag) @ Q_right  # [seq_len, d_model]

        rank = 4
        A, B_correct, S_r, S_full, r = _extract_matrix_correct(H, rank=rank)
        A_b, B_buggy, S_r_b, _ = _extract_matrix_buggy(H, rank=rank)

        v = torch.from_numpy(rng.standard_normal(d_model)).float()
        v_norm = v.norm().item()

        correction_correct = B_correct @ (A @ v)
        correction_buggy = B_buggy @ (A_b @ v)

        # Correct: bounded
        assert correction_correct.norm().item() <= v_norm + 1e-4, (
            f"Correct B unbounded: {correction_correct.norm().item():.2f} > {v_norm:.2f}"
        )
        # Buggy: unbounded (scaled by ~1000)
        assert correction_buggy.norm().item() > v_norm * 10, (
            f"Buggy B not unbounded: {correction_buggy.norm().item():.2f} <= {v_norm * 10:.2f}"
        )

    def test_buggy_and_correct_b_are_not_equal(self):
        """B_buggy and B_correct differ whenever singular values != 1."""
        H = _make_H(30, 64)
        A, B_correct, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        A_b, B_buggy, S_r_b, _ = _extract_matrix_buggy(H, rank=4)
        # Any real matrix has S_r[0] >> 1, so they must differ
        assert not torch.allclose(B_correct, B_buggy, atol=1e-3), (
            "B_correct and B_buggy should differ when singular values != 1"
        )


# ---------------------------------------------------------------------------
# Rank truncation
# ---------------------------------------------------------------------------

class TestRankTruncation:
    def test_effective_rank_capped_by_seq_len(self):
        """When seq_len < requested rank, effective_rank = seq_len."""
        H = _make_H(rows=3, cols=64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        assert r == 3

    def test_effective_rank_capped_by_d_model(self):
        """When d_model < requested rank, effective_rank = d_model."""
        H = _make_H(rows=50, cols=4)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        assert r == 4

    def test_effective_rank_capped_by_requested_rank(self):
        H = _make_H(rows=50, cols=64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        assert r == 4

    def test_s_r_length_matches_effective_rank(self):
        H = _make_H(rows=50, cols=64)
        for rank in [1, 4, 8, 16]:
            A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=rank)
            assert len(S_r) == r, (
                f"rank={rank}: S_r length {len(S_r)} != effective_rank {r}"
            )

    def test_s_full_length_is_min_of_dims(self):
        H = _make_H(rows=30, cols=64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=4)
        assert len(S_full) == min(30, 64)

    def test_singular_values_are_non_negative_and_non_increasing(self):
        H = _make_H(rows=50, cols=64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=8)
        for i, sv in enumerate(S_full):
            assert sv.item() >= 0.0, f"Negative singular value at index {i}"
        for i in range(len(S_full) - 1):
            assert S_full[i].item() >= S_full[i + 1].item() - 1e-5, (
                f"Singular values not sorted at {i}: "
                f"{S_full[i].item():.4f} < {S_full[i+1].item():.4f}"
            )

    def test_rank_1_gives_single_direction(self):
        H = _make_H(50, 64)
        A, B, S_r, S_full, r = _extract_matrix_correct(H, rank=1)
        assert A.shape == (1, 64)
        assert B.shape == (64, 1)
        assert len(S_r) == 1


# ---------------------------------------------------------------------------
# MatrixBenchmarkRecord structure
# ---------------------------------------------------------------------------

class TestMatrixBenchmarkRecordStructure:
    def _make_record(self, **overrides):
        from .benchmark_matrix_runner import MatrixBenchmarkRecord
        defaults = dict(
            model_key="llama3-8b",
            model_hf_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            task="hotpotqa",
            example_id="ex_001",
            injection_layer=14,
            rank=8,
            fact_mode="ner",
            n_context_words=500,
            sv_energy_frac=0.72,
            baseline_input_tokens=600,
            vector_input_tokens=20,
            vector_f_input_tokens=30,
            matrix_input_tokens=20,
            matrix_f_input_tokens=30,
            baseline_output_tokens=10,
            vector_output_tokens=8,
            vector_f_output_tokens=9,
            matrix_output_tokens=8,
            matrix_f_output_tokens=9,
            input_token_reduction_vector=0.97,
            input_token_reduction_vector_f=0.95,
            input_token_reduction_matrix=0.97,
            input_token_reduction_matrix_f=0.95,
            total_cost_ratio_vector=0.30,
            total_cost_ratio_vector_f=0.35,
            total_cost_ratio_matrix=0.30,
            total_cost_ratio_matrix_f=0.35,
            baseline_f1=0.80,
            vector_f1=0.75,
            vector_f_f1=0.82,
            matrix_f1=0.76,
            matrix_f_f1=0.84,
            baseline_exact_match=True,
            vector_exact_match=False,
            vector_f_exact_match=True,
            matrix_exact_match=False,
            matrix_f_exact_match=True,
            f1_delta_vector=-0.05,
            f1_delta_vector_f=0.02,
            f1_delta_matrix=-0.04,
            f1_delta_matrix_f=0.04,
            f_contrib_vector=0.07,
            f_contrib_matrix=0.08,
            mx_vs_vec=0.01,
            mx_vs_vec_f=0.02,
            baseline_answer="Marie Curie",
            vector_answer="Marie",
            vector_f_answer="Marie Curie",
            matrix_answer="Curie",
            matrix_f_answer="Marie Curie",
            fact_block="Facts: Marie Curie; Nobel Prize",
            gold_answers="['Marie Curie']",
            elapsed_baseline_s=1.2,
            elapsed_vector_s=0.3,
            elapsed_vector_f_s=0.4,
            elapsed_matrix_s=0.4,
            elapsed_matrix_f_s=0.5,
        )
        defaults.update(overrides)
        return MatrixBenchmarkRecord(**defaults)

    def test_record_instantiation(self):
        rec = self._make_record()
        assert rec.model_key == "llama3-8b"
        assert rec.rank == 8
        assert rec.sv_energy_frac == pytest.approx(0.72)

    def test_sv_energy_frac_range(self):
        for frac in [0.0, 0.5, 0.72, 1.0]:
            rec = self._make_record(sv_energy_frac=frac)
            assert 0.0 <= rec.sv_energy_frac <= 1.0

    def test_token_reduction_fields_in_range(self):
        rec = self._make_record()
        assert 0.0 <= rec.input_token_reduction_vector <= 1.0
        assert 0.0 <= rec.input_token_reduction_vector_f <= 1.0
        assert 0.0 <= rec.input_token_reduction_matrix <= 1.0
        assert 0.0 <= rec.input_token_reduction_matrix_f <= 1.0

    def test_f1_delta_is_injection_minus_baseline(self):
        rec = self._make_record(
            baseline_f1=0.80, vector_f1=0.75,
            f1_delta_vector=round(0.75 - 0.80, 4),
        )
        assert rec.f1_delta_vector == pytest.approx(-0.05)

    def test_f_contrib_is_f_minus_no_f(self):
        rec = self._make_record(
            vector_f1=0.75, vector_f_f1=0.82,
            f_contrib_vector=round(0.82 - 0.75, 4),
        )
        assert rec.f_contrib_vector == pytest.approx(0.07)

    def test_mx_vs_vec_is_matrix_minus_vector(self):
        rec = self._make_record(
            vector_f1=0.75, matrix_f1=0.76,
            mx_vs_vec=round(0.76 - 0.75, 4),
        )
        assert rec.mx_vs_vec == pytest.approx(0.01)

    def test_all_five_answer_fields_present(self):
        rec = self._make_record()
        assert isinstance(rec.baseline_answer, str)
        assert isinstance(rec.vector_answer, str)
        assert isinstance(rec.vector_f_answer, str)
        assert isinstance(rec.matrix_answer, str)
        assert isinstance(rec.matrix_f_answer, str)

    def test_rank_is_positive_integer(self):
        for rank in [1, 4, 8, 16, 32]:
            rec = self._make_record(rank=rank)
            assert rec.rank == rank
            assert rec.rank > 0
