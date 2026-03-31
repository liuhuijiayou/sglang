"""
Unit tests for the TurboQuant core algorithm (codebook, rotation, quantizer).

Tests verify:
1. Lloyd-Max codebook correctness against paper's closed-form values
2. Rotation matrix orthogonality and determinism
3. End-to-end MSE distortion bounds (Theorem 1 of the paper)
4. Non-unit-norm vector handling
5. GPU consistency

These are pure algorithm tests — no server launch required.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-a-test-1-gpu-small")

import math
import unittest

import torch


class TestCodebook(unittest.TestCase):
    """Tests for turboquant.codebook module."""

    def test_lloyd_max_b1_centroids(self):
        """b=1 centroids should match paper's closed-form: ±sqrt(2/(pi*d))."""
        from sglang.srt.layers.quantization.turboquant.codebook import get_codebook

        d = 128
        centroids, boundaries = get_codebook(d, bits=1)

        expected = math.sqrt(2.0 / (math.pi * d))
        self.assertEqual(centroids.shape[0], 2)
        self.assertAlmostEqual(centroids[0].item(), -expected, places=3)
        self.assertAlmostEqual(centroids[1].item(), expected, places=3)

        # Single boundary at 0
        self.assertEqual(boundaries.shape[0], 1)
        self.assertAlmostEqual(boundaries[0].item(), 0.0, places=5)

    def test_lloyd_max_b2_centroids(self):
        """b=2 centroids should approximately match paper's values for large d."""
        from sglang.srt.layers.quantization.turboquant.codebook import get_codebook

        d = 128
        centroids, boundaries = get_codebook(d, bits=2)

        self.assertEqual(centroids.shape[0], 4)
        # Paper: ±0.453/sqrt(d), ±1.51/sqrt(d)
        sqrt_d = math.sqrt(d)
        expected_inner = 0.453 / sqrt_d
        expected_outer = 1.51 / sqrt_d

        # Centroids should be sorted: [-outer, -inner, inner, outer]
        self.assertAlmostEqual(abs(centroids[0].item()), expected_outer, places=2)
        self.assertAlmostEqual(abs(centroids[1].item()), expected_inner, places=2)
        self.assertAlmostEqual(abs(centroids[2].item()), expected_inner, places=2)
        self.assertAlmostEqual(abs(centroids[3].item()), expected_outer, places=2)

    def test_lloyd_max_convergence(self):
        """Lloyd-Max should converge within max_iter iterations."""
        from sglang.srt.layers.quantization.turboquant.codebook import lloyd_max_solve

        _, _, distortion = lloyd_max_solve(d=128, num_levels=8, max_iter=200)
        self.assertGreater(distortion, 0.0)
        self.assertLess(distortion, 1.0)

    def test_codebook_centroids_sorted(self):
        """Centroids should be sorted in ascending order."""
        from sglang.srt.layers.quantization.turboquant.codebook import get_codebook

        for bits in [1, 2, 3, 4]:
            centroids, boundaries = get_codebook(128, bits)
            for i in range(len(centroids) - 1):
                self.assertLess(centroids[i].item(), centroids[i + 1].item())
            for i in range(len(boundaries) - 1):
                self.assertLess(boundaries[i].item(), boundaries[i + 1].item())


class TestRotation(unittest.TestCase):
    """Tests for turboquant.rotation module."""

    def test_orthogonality(self):
        """R^T R should be identity."""
        from sglang.srt.layers.quantization.turboquant.rotation import (
            get_rotation_matrix,
        )

        R = get_rotation_matrix(128, seed=42)
        eye = torch.eye(128)
        diff = (R.t() @ R - eye).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_determinism(self):
        """Same seed should give same matrix."""
        from sglang.srt.layers.quantization.turboquant.rotation import (
            get_rotation_matrix,
        )

        R1 = get_rotation_matrix(128, seed=123, device="cpu")
        R2 = get_rotation_matrix(128, seed=123, device="cpu")
        self.assertTrue(torch.equal(R1, R2))

    def test_different_seeds(self):
        """Different seeds should give different matrices."""
        from sglang.srt.layers.quantization.turboquant.rotation import (
            get_rotation_matrix,
        )

        R1 = get_rotation_matrix(128, seed=1, device="cpu")
        R2 = get_rotation_matrix(128, seed=2, device="cpu")
        self.assertFalse(torch.equal(R1, R2))

    def test_rotation_preserves_norm(self):
        """Rotation should preserve L2 norm (isometry)."""
        from sglang.srt.layers.quantization.turboquant.rotation import (
            get_rotation_matrix,
            rotate,
        )

        R = get_rotation_matrix(64, seed=42)
        x = torch.randn(32, 64)
        y = rotate(x, R)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        diff = (x_norms - y_norms).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_roundtrip(self):
        """rotate then unrotate should recover original."""
        from sglang.srt.layers.quantization.turboquant.rotation import (
            get_rotation_matrix,
            rotate,
            unrotate,
        )

        R = get_rotation_matrix(64, seed=42)
        x = torch.randn(10, 64)
        y = rotate(x, R)
        x_hat = unrotate(y, R)
        diff = (x - x_hat).abs().max().item()
        self.assertLess(diff, 1e-5)


class TestTurboQuantMSE(unittest.TestCase):
    """Tests for the end-to-end TurboQuant MSE quantizer."""

    def test_compress_decompress_shape(self):
        """Output shapes should match input."""
        from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

        tq = TurboQuantMSE(head_dim=128, bits=3)
        x = torch.randn(4, 8, 128)  # (batch, heads, dim)
        packed, norms = tq.compress(x)

        # 3-bit, head_dim=128 → packed_dim = 128*3/8 = 48
        self.assertEqual(packed.shape, (4, 8, 48))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(norms.shape, (4, 8))
        self.assertEqual(norms.dtype, torch.float16)

        x_hat = tq.decompress(packed, norms)
        self.assertEqual(x_hat.shape, (4, 8, 128))
        self.assertEqual(x_hat.dtype, torch.bfloat16)

    def test_mse_bound_b1(self):
        """MSE distortion should be <= paper's value 0.36 (within tolerance)."""
        self._check_mse_bound(bits=1, paper_mse=0.36, tol=1.1)

    def test_mse_bound_b2(self):
        """MSE distortion should be <= paper's value 0.117 (within tolerance)."""
        self._check_mse_bound(bits=2, paper_mse=0.117, tol=1.1)

    def test_mse_bound_b3(self):
        """MSE distortion should be <= paper's value 0.03 (within tolerance).

        Note: The paper's 0.030 is the d→∞ asymptotic limit.
        For finite d=128, the exact Beta density gives ~0.034,
        which is within the tolerance factor.
        """
        self._check_mse_bound(bits=3, paper_mse=0.03, tol=1.20)

    def test_mse_bound_b4(self):
        """MSE distortion should be <= paper's value 0.009 (within tolerance)."""
        self._check_mse_bound(bits=4, paper_mse=0.009, tol=1.20)

    def _check_mse_bound(self, bits: int, paper_mse: float, tol: float):
        """Helper: check that empirical MSE on random unit vectors matches paper bounds.

        Paper Theorem 1: D_mse <= C for unit-norm vectors on S^{d-1}.
        We generate random unit vectors and verify the average MSE is close to
        the paper's numerically computed values.

        Args:
            bits: bit-width
            paper_mse: expected MSE from paper's Table (Theorem 1)
            tol: multiplicative tolerance (empirical <= paper * tol)
        """
        from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

        d = 128
        n = 5000
        tq = TurboQuantMSE(head_dim=d, bits=bits)

        # Random unit vectors on S^{d-1}
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)

        indices, norms = tq.compress(x)
        x_hat = tq.decompress(indices, norms, out_dtype=torch.float32)

        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        self.assertLess(
            mse,
            paper_mse * tol,
            f"b={bits}: empirical MSE {mse:.4f} exceeds paper bound "
            f"{paper_mse} * {tol} = {paper_mse * tol:.4f}",
        )

    def test_non_unit_norm_roundtrip(self):
        """Vectors with arbitrary norms should be handled correctly."""
        from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

        tq = TurboQuantMSE(head_dim=64, bits=4)
        x = torch.randn(100, 64) * 5.0  # non-unit norm

        indices, norms = tq.compress(x)
        x_hat = tq.decompress(indices, norms, out_dtype=torch.float32)

        # Check norms are approximately preserved
        orig_norms = x.norm(dim=-1)
        recon_norms = x_hat.norm(dim=-1)
        norm_err = ((orig_norms - recon_norms) / orig_norms.clamp(min=1e-6)).abs()
        self.assertLess(
            norm_err.mean().item(), 0.15, "Norm preservation error too large"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_compress_gpu(self):
        """GPU decompress quality should match CPU (indices may differ due to bf16 rotation)."""
        from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

        d, bits = 128, 3
        tq_cpu = TurboQuantMSE(head_dim=d, bits=bits, device="cpu")
        tq_gpu = TurboQuantMSE(head_dim=d, bits=bits, device="cuda")

        x_cpu = torch.randn(32, d)
        x_gpu = x_cpu.cuda()

        packed_cpu, norm_cpu = tq_cpu.compress(x_cpu)
        packed_gpu, norm_gpu = tq_gpu.compress(x_gpu)

        # Decompress and compare reconstruction quality (not exact indices,
        # because GPU uses bf16 rotation which may flip indices near boundaries)
        x_hat_cpu = tq_cpu.decompress(packed_cpu, norm_cpu, out_dtype=torch.float32)
        x_hat_gpu = tq_gpu.decompress(packed_gpu, norm_gpu, out_dtype=torch.float32)

        # Both should achieve similar MSE vs original
        mse_cpu = (x_cpu - x_hat_cpu).pow(2).sum(dim=-1).mean().item()
        mse_gpu = (x_cpu - x_hat_gpu.cpu()).pow(2).sum(dim=-1).mean().item()
        # MSE should be within 20% of each other
        ratio = max(mse_cpu, mse_gpu) / max(min(mse_cpu, mse_gpu), 1e-10)
        self.assertLess(ratio, 1.2, f"CPU MSE={mse_cpu:.4f}, GPU MSE={mse_gpu:.4f}")

    def test_zero_vector(self):
        """Zero vectors should not cause NaN or crash."""
        from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

        tq = TurboQuantMSE(head_dim=64, bits=2)
        x = torch.zeros(5, 64)
        indices, norms = tq.compress(x)
        x_hat = tq.decompress(indices, norms, out_dtype=torch.float32)
        self.assertFalse(torch.isnan(x_hat).any())


if __name__ == "__main__":
    unittest.main()
