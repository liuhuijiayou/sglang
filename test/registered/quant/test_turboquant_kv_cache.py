"""
Integration test for TurboQuant KV cache compression.

Launches an SGLang server with --kv-cache-dtype turboquant_3bit and runs
a few-shot GSM8K evaluation to verify end-to-end correctness.

Follows the same pattern as test_fp8kv_triton.py.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-test-1-gpu-large")

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTurboQuantKVCache3Bit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-dtype",
                "turboquant_3bit",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=200,
            host=f"{parsed_url.scheme}://{parsed_url.hostname}",
            port=parsed_url.port,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # Use a relaxed threshold for the MVP. TurboQuant 3-bit on a small model
        # may have noticeable quality impact. The primary goal is to verify the
        # feature doesn't crash and produces reasonable outputs.
        self.assertGreater(metrics["accuracy"], 0.40)


class TestTurboQuantKVCache4Bit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-dtype",
                "turboquant_4bit",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=200,
            host=f"{parsed_url.scheme}://{parsed_url.hostname}",
            port=parsed_url.port,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # 4-bit should be closer to full precision
        self.assertGreater(metrics["accuracy"], 0.50)


if __name__ == "__main__":
    unittest.main()
