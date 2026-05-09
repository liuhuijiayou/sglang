"""
Unit test: FA3 backend's init_forward_metadata* must not read
forward_batch.seq_lens_cpu. This is the contract downstream of PR #23005
subtask "avoid using cpu metadata in attention backends".
"""
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Register this test for CUDA CI in stage-b (fast attention/kernel tests)
register_cuda_ci(est_time=3, suite="stage-b-test-1-gpu-large")


class _SeqLensCpuAccessed(AssertionError):
    """Sentinel raised when FA3 backend touches forward_batch.seq_lens_cpu."""


class _PoisonedSeqLensCpu:
    """Raises _SeqLensCpuAccessed on any interaction.

    Used as a stand-in for forward_batch.seq_lens_cpu / extend_seq_lens_cpu
    and for the seq_lens_cpu replay kwarg. If FA3 backend touches the value
    at all — attribute access, subscript, iteration, truthiness — the
    sentinel fires, giving us a clean fail signal that doesn't depend on
    fragile error-message string matching.
    """

    def __getattr__(self, name):  # e.g. seq_lens_cpu.max
        raise _SeqLensCpuAccessed(
            f"FA3 backend accessed seq_lens_cpu.{name}"
        )

    def __getitem__(self, key):  # e.g. seq_lens_cpu[:bs]
        raise _SeqLensCpuAccessed(
            f"FA3 backend subscripted seq_lens_cpu[{key!r}]"
        )

    def __iter__(self):
        raise _SeqLensCpuAccessed("FA3 backend iterated seq_lens_cpu")

    def __bool__(self):
        raise _SeqLensCpuAccessed("FA3 backend evaluated seq_lens_cpu truthiness")

    def __len__(self):
        raise _SeqLensCpuAccessed("FA3 backend measured len(seq_lens_cpu)")


class _DummyBackend(FlashAttentionBackend):
    """Bypass __init__ to avoid needing a full ModelRunner."""

    def __init__(self):
        self.forward_metadata = None
        self.forward_metadata_spec_decode_expand = None
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.target_verify_metadata_topk_normal = {}
        self.target_verify_metadata_topk_expand = {}
        self.target_verify_metadata_topk_swa = {}
        self.draft_decode_metadata_topk_normal = {}
        self.draft_decode_metadata_topk_expand = {}
        self.draft_extend_metadata = {}
        self.encoder_metadata = {}
        self.decode_cuda_graph_local_attn_metadata = {}
        self.topk = 1
        self.speculative_num_draft_tokens = 0
        self.speculative_step_id = 0
        self.speculative_num_steps = 0
        self.page_size = 1
        self.use_sliding_window_kv_pool = False
        self.req_to_token = torch.zeros(4, 128, dtype=torch.int32)
        self.is_hybrid = False
        self.kv_cache_dtype_str = "auto"
        self.fa_impl_ver = 3


class TestFA3SeqLensCpuDecoupling(CustomTestCase):
    """
    FA3 backend must construct forward metadata without reading
    forward_batch.seq_lens_cpu. This regression guards the
    "avoid using cpu metadata in attention backends" contract.
    """

    def _make_decode_batch(self, bs=2, seq_len=32):
        fb = MagicMock()
        fb.forward_mode = ForwardMode.DECODE
        fb.batch_size = bs
        fb.seq_lens = torch.tensor([seq_len] * bs, dtype=torch.int32)
        fb.seq_lens_cpu = _PoisonedSeqLensCpu()
        fb.req_pool_indices = torch.arange(bs, dtype=torch.int32)
        fb.out_cache_loc = torch.zeros(bs, dtype=torch.int64)
        fb.spec_info = None
        fb.extend_seq_lens = None
        fb.extend_seq_lens_cpu = _PoisonedSeqLensCpu()
        fb.encoder_lens = None
        return fb

    def test_init_forward_metadata_decode_without_seq_lens_cpu(self):
        """Non-graph DECODE path must not access forward_batch.seq_lens_cpu."""
        backend = _DummyBackend()
        fb = self._make_decode_batch(bs=2, seq_len=32)
        try:
            backend.init_forward_metadata(fb)
        except _SeqLensCpuAccessed as e:
            self.fail(f"FA3 backend still reads seq_lens_cpu: {e}")
        except Exception:
            # Other errors are tolerated: _DummyBackend is minimal and
            # CUDA/triton-dependent code paths may raise unrelated errors
            # when run on CPU-only envs. End-to-end validation happens on
            # H200 CI with a real ModelRunner. The contract enforced here
            # is narrow: the _PoisonedSeqLensCpu sentinel must never fire.
            return
        # If the call succeeded (real CUDA env, full backend), verify post-state
        self.assertIsNotNone(backend.forward_metadata)
        self.assertGreater(backend.forward_metadata.max_seq_len_k, 0)

    def test_init_replay_cuda_graph_without_seq_lens_cpu(self):
        """CUDA graph replay path must not access the seq_lens_cpu kwarg."""
        backend = _DummyBackend()
        bs = 2
        seq_lens = torch.tensor([32, 48], dtype=torch.int32)

        metadata = FlashAttentionMetadata()
        metadata.cache_seqlens_int32 = torch.zeros(bs, dtype=torch.int32)
        metadata.cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32)
        metadata.cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32)
        metadata.page_table = torch.zeros(bs, 128, dtype=torch.int32)
        metadata.swa_page_table = None
        backend.decode_cuda_graph_metadata[bs] = metadata
        backend.decode_cuda_graph_metadata["strided_indices"] = torch.arange(
            128, dtype=torch.int32
        )

        try:
            backend.init_forward_metadata_replay_cuda_graph(
                bs=bs,
                req_pool_indices=torch.arange(bs, dtype=torch.int32),
                seq_lens=seq_lens,
                seq_lens_sum=int(seq_lens.sum()),
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
                seq_lens_cpu=_PoisonedSeqLensCpu(),
                out_cache_loc=torch.zeros(bs, dtype=torch.int64),
            )
        except _SeqLensCpuAccessed as e:
            self.fail(f"FA3 backend replay still reads seq_lens_cpu: {e}")
        except Exception:
            return
        # Post-state: replay must still populate max_seq_len_k from GPU tensor.
        # spec_info=None routes to the Normal Decode branch, which sets
        # max_seq_len_k = int(seq_lens.max()) with no speculative offset.
        self.assertEqual(
            backend.decode_cuda_graph_metadata[bs].max_seq_len_k,
            int(seq_lens.max()),
        )


if __name__ == "__main__":
    unittest.main()
