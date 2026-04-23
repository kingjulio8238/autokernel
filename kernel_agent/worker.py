# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verification Worker for testing and refining individual kernels."""

import ast
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from kernel_agent.platform_config import get_platform
from kernel_agent.worker_util import format_test_code_for_llm
from kernel_agent.ka_utils.providers import get_model_provider

from .prompt_manager import PromptManager
from .worker_util import _run_test_multiprocess


# LLMs occasionally emit typographic Unicode (smart quotes, em-dash, bullets)
# inside generated code, which makes Python's parser fail immediately with
# "SyntaxError: invalid character …" even when the surrounding code is fine.
# We normalize on both the Modal-side write (modal_infra/app.py) AND here
# so the local subprocess eval path gets the same treatment.
_UNICODE_CODE_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‐": "-", "‑": "-",
    "‒": "-", "−": "-", "…": "...", " ": " ",
    "•": "#", "·": "#", "﻿": "",
}


def _sanitize_kernel_source(src: str) -> str:
    """Replace typographic Unicode with ASCII equivalents. Idempotent."""
    if not src:
        return src
    for bad, good in _UNICODE_CODE_REPLACEMENTS.items():
        if bad in src:
            src = src.replace(bad, good)
    return src


DISALLOWED_TORCH_PATTERNS = [
    # REMOVED: `import torch.nn` / `from torch import nn` / `torch.nn.Module`
    # / `torch.nn.Parameter` — these are STRUCTURALLY REQUIRED by the
    # KernelBench format (Model / ModelNew subclass nn.Module, declare
    # nn.Parameter fields, etc.). The original patterns were over-strict
    # and rejected every KB-format kernel. The real cheat patterns below
    # (torch.nn.functional.*, torch.matmul/mm/bmm, torch.relu/etc., F.*(),
    # torch.conv, torch.einsum, torch.ops.aten) still catch the things
    # that would trivially bypass a Triton implementation.
    (
        re.compile(r"\bimport\s+torch\.nn\.functional\s+as\s+F\b"),
        "aliasing torch.nn.functional as F is not allowed",
    ),
    (
        re.compile(r"\btorch\.nn\.functional\b"),
        "torch.nn.functional usage is not allowed",
    ),
    (
        re.compile(r"\bF\.[A-Za-z_]+\("),
        "torch.nn.functional alias calls (F.*) are not allowed",
    ),
    (re.compile(r"\btorch\.conv"), "torch convolution helpers are not allowed"),
    (
        re.compile(
            r"\btorch\.(relu|sigmoid|tanh|softmax|gelu|mish|hardtanh|max_pool|avg_pool)[A-Za-z0-9_]*\("
        ),
        "PyTorch activation/pooling helpers are not allowed",
    ),
    # REMOVED: `class X(nn.Module)` and `.forward(` patterns — these are
    # structurally REQUIRED by KernelBench format (harness instantiates
    # `ModelNew()` and calls it, which invokes forward via __call__). The
    # original validator was written for gpumode-only workflow where no
    # ModelNew class exists. See path-c-validation report for evidence.
    (
        re.compile(r"\btorch\.ops\.aten\b"),
        "Low-level torch.ops.aten.* calls are not allowed; implement these ops directly in Triton kernels instead of relying on PyTorch compute",
    ),
    # Generic tensor-tensor math that must be implemented in Triton kernels
    (
        re.compile(r"\btorch\.(matmul|mm|bmm)\s*\("),
        "PyTorch matmul/mm/bmm tensor-tensor ops are not allowed; implement these in Triton kernels",
    ),
    (
        re.compile(r"\.(matmul|mm|bmm)\s*\("),
        "Tensor.matmul/mm/bmm methods are not allowed; implement these in Triton kernels",
    ),
    (
        re.compile(r"\btorch\.einsum\s*\("),
        "torch.einsum is not allowed; implement this contraction with Triton primitives",
    ),
    (
        re.compile(r"\.einsum\s*\("),
        "Tensor.einsum is not allowed; implement this contraction with Triton primitives",
    ),
    # Introspection / frame inspection that can be used to steal test locals
    (
        re.compile(r"\bimport\s+inspect\b"),
        "inspect-based reflection is not allowed inside kernel files",
    ),
    (
        re.compile(r"\binspect\.(stack|currentframe|getouterframes)\s*\("),
        "inspect stack/frame introspection is not allowed in kernels",
    ),
    (
        re.compile(r"\bsys\._getframe\s*\("),
        "sys._getframe is not allowed in kernels; do not access caller frames",
    ),
    (
        re.compile(r"\.f_locals\b|\.f_globals\b"),
        "Accessing frame locals/globals (f_locals/f_globals) from kernels is not allowed",
    ),
    (
        re.compile(r"\bglobals\s*\("),
        "globals() is not allowed in kernels; avoid depending on ambient test state",
    ),
    (
        re.compile(r"\blocals\s*\("),
        "locals() is not allowed in kernels; avoid depending on caller scopes",
    ),
]


class VerificationWorker:
    """Worker that verifies and refines a single kernel implementation."""

    def __init__(
        self,
        worker_id: int,
        workdir: Path,
        log_dir: Path,
        max_rounds: int = 10,
        history_size: int = 8,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-5",
        high_reasoning_effort: bool = True,
        target_platform: str = "cuda",
        no_cusolver: bool = False,
        test_timeout_s: int = 30,
    ):
        """
        Initialize a verification worker.

        Args:
            worker_id: Unique identifier for this worker
            workdir: Working directory for this worker
            log_dir: Directory for logging
            max_rounds: Maximum refinement rounds
            history_size: Number of recent rounds to keep
            openai_api_key: OpenAI API key for refinement
            openai_model: Model name for refinement
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
            target_platform: Target platform default: cuda
            no_cusolver: If True, disables cuSolver library usage
            test_timeout_s: Timeout in seconds for test execution
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.history_size = history_size
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self._platform_config = get_platform(target_platform)
        self.no_cusolver = no_cusolver
        self.test_timeout_s = test_timeout_s

        # Remote eval via Modal — detected from env var OPENKERNEL_USE_MODAL=1
        # When set, _run_test() calls Modal's eval_kernel_on_gpu instead of subprocess
        self._use_modal = os.environ.get("OPENKERNEL_USE_MODAL") == "1"
        self._reference_code = os.environ.get("OPENKERNEL_REFERENCE_CODE", "")

        # Progress callbacks (only work in-process, not across multiprocessing)
        self._on_phase = None
        self._on_result = None

        # Setup files
        self.kernel_file = self.workdir / "kernel.py"
        self.test_files: list[Path] = []

        # History for LLM context
        self.history = deque(maxlen=history_size)

        # Dev logging: last LLM prompt/response for thought capture
        self._last_prompt: str = ""
        self._last_response: str = ""

        # Setup logging early so it is available for any error paths
        self._setup_logging()

        # Initialize prompt manager with resolved config
        self.prompt_manager = PromptManager(target_platform=self._platform_config)

        # Initialize provider (may be unavailable in offline/test environments)
        self.provider = None
        try:
            self.provider = get_model_provider(self.openai_model)
        except ValueError as e:
            # Provider not available, will use mock mode
            self.logger.warning(f"Provider not available: {e}")

    def _setup_logging(self):
        """Setup worker-specific logging."""
        log_file = self.log_dir / f"worker_{self.worker_id}.log"
        self.logger = logging.getLogger(f"worker_{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def _extract_code_from_response(
        self,
        response_text: str,
        language: str = "python",
        prefer_kernel_function: bool = False,
    ) -> str | None:
        """
        Extract code from LLM response text.

        Args:
            response_text: The full LLM response text
            language: The expected language (default: python)
            prefer_kernel_function: When True and multiple code blocks are
                found, prefer the block that defines ``kernel_function``
                (falling back to the longest block).  Use this when the
                prompt contains additional test code that the LLM may echo
                back.

        Returns:
            Extracted code or None if no valid code block found
        """
        if not response_text:
            return None

        # First, try to find code blocks with language markers
        # Pattern matches ```python or ```language_name
        pattern = rf"```{language}\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            # Try generic code blocks without language marker
            pattern = r"```\s*\n(.*?)```"
            matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            if prefer_kernel_function and len(matches) > 1:
                # When additional tests are in the prompt the LLM may echo
                # wrapper code.  Prefer the block defining kernel_function.
                for block in matches:
                    if re.search(r"\bdef\s+kernel_function\b", block):
                        return block.strip()
                # Fallback: return the longest block
                return max(matches, key=len).strip()
            # Default: return the first match
            return matches[0].strip()

        # If no code blocks found, check if the entire response looks like code
        # This is a fallback for cases where LLM doesn't use code blocks
        lines = response_text.strip().split("\n")

        # Simple heuristic: if response contains import statements or function definitions
        code_indicators = ["import ", "from ", "def ", "class ", "@", '"""', "'''"]
        if any(
            line.strip().startswith(indicator)
            for line in lines
            for indicator in code_indicators
        ):
            # Likely the entire response is code
            return response_text.strip()

        # No code found
        self.logger.warning("No code block found in LLM response")
        return None

    def _write_kernel(self, kernel_code: str):
        """Write only the kernel code to file."""
        self.kernel_file.write_text(_sanitize_kernel_source(kernel_code))
        self.logger.info("Updated kernel file")

    def _write_files(self, kernel_code: str, test_code: list[str]):
        """Write kernel and test code to files.

        Note: The test code should import the kernel function from the kernel file:
            from kernel import kernel_function

        Both files are written to the same directory (workdir).

        Args:
            kernel_code: The kernel source code.
            test_code: List of test code strings. ``test_code[0]`` is the
                primary test written to ``test_kernel.py``; any subsequent
                entries are written to ``test_extra_{i}_kernel.py``.
        """
        self.kernel_file.write_text(_sanitize_kernel_source(kernel_code))
        self.test_files = []
        for i, code in enumerate(test_code):
            name = "test_kernel.py" if i == 0 else f"test_extra_{i}_kernel.py"
            path = self.workdir / name
            path.write_text(code)
            self.test_files.append(path)
        self.logger.info("Wrote kernel and %d test file(s)", len(self.test_files))

    def _strip_comments_and_strings(self, code: str) -> str:
        """Remove comments, docstrings, and ``if __name__ == "__main__":``
        blocks so the scanner doesn't false-positive on code that never
        executes at eval time.

        - Preserves newlines inside multi-line docstrings so line numbers
          still map to the original source.
        - Drops anything inside an ``if __name__ == "__main__":`` block
          (replaced with blank lines to preserve numbering). The Modal
          eval harness imports the kernel module — Python does not run
          ``__main__`` blocks on import, so patterns there are harmless
          and must not block a kernel that is otherwise clean.
        """
        pattern = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#.*)')

        def _blank_but_keep_newlines(m: "re.Match[str]") -> str:
            matched = m.group(0)
            if matched.startswith("#"):
                return ""
            return re.sub(r"[^\n]", " ", matched)

        stripped = pattern.sub(_blank_but_keep_newlines, code)

        # Blank out ``if __name__ == "__main__":`` + body. The block starts
        # at column 0 and runs until EOF or the next unindented line.
        main_re = re.compile(
            r'^if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*$',
            re.MULTILINE,
        )
        m = main_re.search(stripped)
        if m:
            head = stripped[: m.start()]
            tail = stripped[m.start():]
            # Replace every non-newline char in the tail with space so the
            # indented body is wiped but line numbers still match.
            tail_blanked = re.sub(r"[^\n]", " ", tail)
            stripped = head + tail_blanked

        return stripped

    def _detect_pytorch_compute(self, kernel_code: str) -> str | None:
        """Detect disallowed PyTorch usage inside the kernel wrapper.

        Returns a message that includes the offending line number + the raw
        source line (not the comment-stripped version) so the next round's
        LLM gets a concrete pointer to what it wrote. Generic "matmul not
        allowed" messages produce the same offending kernel on every retry
        because the LLM can't tell which line to change.

        Pre-flight: if the kernel doesn't parse (SyntaxError) we short-
        circuit with a parse-error message BEFORE scanning for forbidden
        patterns. Truncated LLM responses frequently leave an unclosed
        triple-quoted docstring; without this guard the scanner sees the
        DOCSTRING PROSE as live code and false-positives on phrases like
        "torch.nn.functional" that are quoting our own prompt back at us.
        """
        try:
            ast.parse(kernel_code)
        except SyntaxError as exc:
            return (
                f"Generated file does not parse as Python — "
                f"regenerate a COMPLETE kernel file. "
                f"Python says: {exc.msg} (line {exc.lineno or '?'})"
            )

        sanitized = self._strip_comments_and_strings(kernel_code)
        for pattern, message in DISALLOWED_TORCH_PATTERNS:
            m = pattern.search(sanitized)
            if not m:
                continue
            line_no = sanitized.count("\n", 0, m.start()) + 1
            src_lines = kernel_code.splitlines()
            if 1 <= line_no <= len(src_lines):
                offending = src_lines[line_no - 1].strip()
                return f"{message} — line {line_no}: {offending[:160]}"
            return message
        return None

    def _run_test(self) -> tuple[bool, str, str]:
        """
        Run all test scripts sequentially (``&&`` semantics).

        If a remote eval function is configured (e.g. Modal), uses that
        instead of local subprocess. This allows running LLM generation
        on CPU and kernel eval on remote GPU.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # --- Remote eval path (Modal) ---
        if self._use_modal:
            return self._run_remote_eval()

        # --- Local eval path (original) ---
        try:
            for test_file in self.test_files:
                if not test_file.exists():
                    continue
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout_s,
                )
                if result.returncode != 0:
                    self.logger.error(
                        "Test %s failed. Exit code: %s, stderr: %s",
                        test_file.name,
                        result.returncode,
                        result.stderr[:2000],
                    )
                    return False, result.stdout, result.stderr
                self.logger.info("Test %s passed", test_file.name)

            return True, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error("Test timed out")
            return (
                False,
                "",
                f"Test execution timed out after {self.test_timeout_s} seconds",
            )
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            return False, "", str(e)

    def _run_remote_eval(self) -> tuple[bool, str, str]:
        """Run kernel evaluation via Modal remote GPU.

        The worker reads OPENKERNEL_REFERENCE_CODE from env (set by the bridge)
        and calls Modal's eval_kernel_on_gpu function. This works across
        multiprocessing boundaries since it uses env vars + Modal's remote API.
        """
        try:
            import modal
            from kernel_agent.model_wrapper import wrap_in_model_new

            kernel_code = self.kernel_file.read_text()
            reference_code = self._reference_code or os.environ.get("OPENKERNEL_REFERENCE_CODE", "")

            if not reference_code:
                return False, "", "No reference code available for remote eval"

            # Determine format and wrap if needed
            problem_format = os.environ.get("OPENKERNEL_PROBLEM_FORMAT", "auto")
            if problem_format == "kernelbench" or "class Model" in reference_code:
                # KernelBench: wrap in ModelNew
                wrapped_code = wrap_in_model_new(kernel_code, reference_code)
            else:
                # GPU Mode: kernel_function() is already the right format
                wrapped_code = kernel_code

            # Route to GPU-specific Modal function based on hardware setting
            from kernel_code.gpu_functions import GPU_FUNCTION_MAP
            gpu_type = os.environ.get("OPENKERNEL_GPU_TYPE", "L40S")
            fn_name = GPU_FUNCTION_MAP.get(gpu_type, "eval_kernel_on_gpu")
            eval_fn = modal.Function.from_name("openkernel-eval", fn_name)
            result = eval_fn.remote(
                kernel_source=wrapped_code,
                reference_source=reference_code,
                eval_mode="fast",
                problem_format=problem_format,
                gpu_type=gpu_type,
            )

            correct = result.get("correct", False)
            speedup = result.get("speedup", 0.0)
            error = result.get("error", "")

            if correct:
                stdout = f"PASS\nSpeedup: {speedup:.4f}x"
                self.logger.info("Remote eval PASS: %.2fx speedup", speedup)
                return True, stdout, ""
            else:
                stderr = error or f"Incorrect result (speedup: {speedup:.4f}x)"
                self.logger.info("Remote eval FAIL: %s", stderr[:200])
                return False, "", stderr

        except Exception as e:
            self.logger.error("Remote eval error: %s", e)
            return False, "", str(e)

    def _call_llm(self, messages: list, **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Captures prompt and response on instance for dev logging.
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.openai_model}")

        # Capture prompt for dev logging
        self._last_prompt = messages[-1].get("content", "") if messages else ""

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.openai_model, messages, **kwargs)
        self._last_response = response.content
        return response.content

    def _refine_kernel(
        self,
        kernel_code: str,
        error_info: dict[str, str],
        problem_description: str,
        test_code: str,
    ) -> str:
        """
        Refine kernel based on error information using OpenAI API.

        Uses multi-turn dialogue by incorporating history of previous attempts.
        """
        if self.provider:
            try:
                self.logger.info(f"Refining kernel using {self.openai_model}")

                # Build context from history — bounded to the LAST 4 attempts
                # with per-attempt caps. Unbounded history squeezed the LLM's
                # output token budget (Conv2D: kernels shrank 209→82→17 lines
                # as the prompt grew across rounds). Last-2 was too tight:
                # workers lost the signal and produced the SAME buggy kernel
                # across retries. Last-4 + ~1000-char stderr budget balances:
                # the LLM sees enough error-context to iterate, without so
                # much history that the output gets truncated.
                history_context = ""
                recent = self.history[-4:] if self.history else []
                if recent:
                    history_context = "\n\nRECENT ATTEMPTS (most recent last):\n"
                    base = len(self.history) - len(recent) + 1
                    for i, round_data in enumerate(recent):
                        history_context += f"\nAttempt {base + i}:\n"
                        history_context += f"Kernel code (head):\n```python\n{round_data['kernel_code'][:400]}\n```\n"
                        if round_data.get("stderr"):
                            history_context += f"Error: {round_data['stderr'][:1000]}\n"
                        if round_data.get("stdout"):
                            history_context += f"Output: {round_data['stdout'][:400]}\n"

                # Create refinement prompt using template
                prompt = self.prompt_manager.render_kernel_refinement_prompt(
                    problem_description=problem_description,
                    test_code=test_code,
                    kernel_code=kernel_code,
                    error_info=error_info,
                    history_context=history_context,
                    no_cusolver=self.no_cusolver,
                )

                # Call LLM API
                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_llm(messages, max_tokens=8192)

                # Store for dev logging
                self._last_prompt = prompt
                self._last_response = response_text

                # Extract refined kernel from response
                refined_kernel = self._extract_code_from_response(
                    response_text,
                    prefer_kernel_function=getattr(self, "_has_multiple_tests", False),
                )

                if refined_kernel:
                    self.logger.info(
                        f"Successfully refined kernel using {self.openai_model}"
                    )
                    return refined_kernel
                else:
                    self.logger.error("Failed to extract valid code from LLM response")
                    # Return original kernel if extraction fails
                    return kernel_code

            except Exception as e:
                self.logger.error(f"Error refining kernel with LLM API: {e}")
                # Fall back to mock refinement

        # Mock refinement (fallback)
        self.logger.info("Refining kernel (mock implementation)")

        # For testing, make a simple modification
        if "error" in error_info.get("stderr", "").lower():
            # Add a comment to show refinement happened
            return f"# Refinement attempt {len(self.history) + 1}\n{kernel_code}"

        return kernel_code

    def _log_round(
        self, round_num: int, success: bool, kernel_code: str, stdout: str, stderr: str,
        prompt: str = "", response: str = "",
    ):
        """Log the results of a verification round.

        In dev mode (OPENKERNEL_DEV_LOG=1), also saves full LLM prompt
        and response for analysis.
        """
        round_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "kernel_code": kernel_code,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Dev mode: append full LLM thoughts to per-run log file
        if os.environ.get("OPENKERNEL_DEV_LOG") == "1" and (prompt or response):
            run_id = os.environ.get("OPENKERNEL_RUN_ID", "unknown")
            log_file = Path(f".kernel-code/dev_logs/run_{run_id}.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            thought_data = {
                "type": "worker",
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "worker_id": self.worker_id,
                "model": self.openai_model,
                "prompt": prompt,
                "response": response,
                "extracted_kernel_length": len(kernel_code) if kernel_code else 0,
                "success": success,
            }
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(thought_data) + "\n")
            except Exception:
                pass  # Don't break optimization for logging

        # Save to log file
        round_log_file = self.log_dir / f"round_{round_num}.json"
        with open(round_log_file, "w") as f:
            json.dump(round_data, f, indent=2)

        # Add to history
        self.history.append(round_data)

    def run(
        self,
        kernel_code: str,
        test_code: list[str],
        problem_description: str,
        success_event: mp.Event,
    ) -> dict[str, Any]:
        """
        Run verification and refinement loop.

        Args:
            kernel_code: Initial kernel implementation
            test_code: List of test code strings (primary + additional tests)
            problem_description: Problem description for context
            success_event: Shared event to check if another worker succeeded

        Returns:
            Dictionary with results
        """
        self.logger.info(f"Starting verification for worker {self.worker_id}")
        self._has_multiple_tests = len(test_code) > 1

        current_kernel = kernel_code

        for round_num in range(self.max_rounds):
            # Check if another worker has succeeded
            if success_event.is_set():
                self.logger.info("Another worker succeeded, stopping")
                return {
                    "worker_id": self.worker_id,
                    "success": False,
                    "stopped_early": True,
                    "rounds": round_num,
                }

            self.logger.info(f"Round {round_num + 1}/{self.max_rounds}")

            # Progress callback
            if self._on_phase:
                self._on_phase(
                    f"Worker {self.worker_id}: evaluating kernel "
                    f"(round {round_num + 1}/{self.max_rounds})"
                )

            # Write files - test only on first round, kernel every round
            if round_num == 0:
                # First round: write both kernel and test(s)
                self._write_files(current_kernel, test_code)
            else:
                # Subsequent rounds: only update kernel, test remains unchanged
                self._write_kernel(current_kernel)

            # Run verification (additional tests chained automatically by _run_test)
            success, stdout, stderr, violation = self._single_verification_pass(
                current_kernel
            )

            if violation:
                self._log_round(round_num + 1, False, current_kernel, "", violation,
                                prompt=self._last_prompt, response=self._last_response)
                error_info = {
                    "stdout": "",
                    "stderr": violation,
                    "history": list(self.history),
                }
                current_kernel = self._refine_kernel(
                    current_kernel,
                    error_info,
                    problem_description,
                    format_test_code_for_llm(test_code),
                )
                continue

            # Log round
            self._log_round(round_num + 1, success, current_kernel, stdout, stderr,
                            prompt=self._last_prompt, response=self._last_response)

            if success:
                self.logger.info(
                    f"Success! Kernel passed test in round {round_num + 1}"
                )
                if self._on_result:
                    self._on_result(round_num + 1, True, 0.0, "")
                return {
                    "worker_id": self.worker_id,
                    "success": True,
                    "kernel_code": current_kernel,
                    "rounds": round_num + 1,
                    "history": list(self.history),
                }

            # Report failure
            if self._on_result:
                err_msg = stderr[:100] if stderr else "verification failed"
                self._on_result(round_num + 1, False, 0.0, err_msg)

            # Refine kernel for next round
            if self._on_phase:
                self._on_phase(
                    f"Worker {self.worker_id}: refining kernel "
                    f"(round {round_num + 1}/{self.max_rounds})"
                )

            error_info = {
                "stdout": stdout,
                "stderr": stderr,
                "history": list(self.history),
            }

            current_kernel = self._refine_kernel(
                current_kernel,
                error_info,
                problem_description,
                format_test_code_for_llm(test_code),
            )

        # Max rounds reached without success
        self.logger.warning(f"Max rounds ({self.max_rounds}) reached without success")
        return {
            "worker_id": self.worker_id,
            "success": False,
            "max_rounds_reached": True,
            "rounds": self.max_rounds,
            "history": list(self.history),
        }

    def _single_verification_pass(
        self, kernel_code: str
    ) -> tuple[bool, str, str, str | None]:
        """
        Run a single verification pass on the kernel.

        Returns:
            Tuple of (success, stdout, stderr, violation_message)
            - violation_message is set if PyTorch usage detected, None otherwise
        """
        violation = self._detect_pytorch_compute(kernel_code)
        if violation:
            message = f"Disallowed PyTorch usage detected: {violation}"
            self.logger.error(message)
            return False, "", message, message

        success, stdout, stderr = (
            self._run_test()
            if os.getenv("KA_PROCESS_USE_SYS_EXECUTABLE", "1") == "1"
            else _run_test_multiprocess(
                self.logger,
                self.workdir,
                self.test_files,
            )
        )

        return success, stdout, stderr, None

    def verify_with_refinement(
        self,
        kernel_code: str,
        test_code: list[str],
        problem_description: str,
        max_refine_attempts: int = 3,
    ) -> tuple[bool, str, str]:
        """
        Verify kernel correctness with refinement attempts.

        This is a simpler API for single-pass verification with refinement,
        useful for optimization loops that manage their own iteration.

        Args:
            kernel_code: Kernel code to verify
            test_code: List of test code strings (primary + additional tests)
            problem_description: Problem description for refinement context
            max_refine_attempts: Maximum refinement attempts if verification fails

        Returns:
            Tuple of (success, final_kernel_code, error_feedback)
            - success: Whether the kernel passed verification
            - final_kernel_code: The verified (possibly refined) kernel
            - error_feedback: Error message if failed, empty string if success
        """
        current_kernel = kernel_code
        self._has_multiple_tests = len(test_code) > 1

        # Write files for testing (primary + additional tests)
        self._write_files(current_kernel, test_code)

        # Initial verification (additional tests chained automatically by _run_test)
        success, stdout, stderr, violation = self._single_verification_pass(
            current_kernel
        )

        if violation:
            # Log initial failure so refinement LLM sees it in history
            self._log_round(0, False, current_kernel, stdout, stderr)
            return False, current_kernel, violation

        if success:
            self.logger.info("✅ Verification passed on first attempt")
            return True, current_kernel, ""

        # Refinement loop
        for attempt in range(1, max_refine_attempts + 1):
            error_output = stderr if stderr.strip() else stdout
            self.logger.info(f"Refinement attempt {attempt}/{max_refine_attempts}...")

            error_info = {
                "stdout": stdout,
                "stderr": stderr,
                "error_type": (
                    "compilation"
                    if "CompilationError" in error_output
                    or "SyntaxError" in error_output
                    else "runtime"
                ),
            }

            # Refine kernel
            refined_kernel = self._refine_kernel(
                current_kernel,
                error_info,
                problem_description,
                format_test_code_for_llm(test_code),
            )

            # Write and test refined kernel
            self._write_kernel(refined_kernel)
            success, stdout, stderr, violation = self._single_verification_pass(
                refined_kernel
            )

            if violation:
                current_kernel = refined_kernel
                continue

            if success:
                self.logger.info(
                    f"✅ Verification passed after refinement (attempt {attempt})"
                )
                return True, refined_kernel, ""

            current_kernel = refined_kernel

        # All attempts exhausted
        error_output = stderr if stderr.strip() else stdout
        error_feedback = f"Verification failed after {max_refine_attempts} refinement attempts:\n{error_output[:2000]}"
        self.logger.warning(f"❌ {error_feedback[:200]}...")
        return False, current_kernel, error_feedback
