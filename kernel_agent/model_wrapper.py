"""Wrap KernelAgent output in KernelBench ModelNew format.

KernelAgent outputs ``kernel_function(...)`` (a plain function).
KernelBench expects ``class ModelNew(nn.Module)`` with ``forward()``.

This module converts between the two formats.
"""

from __future__ import annotations

import ast
import re


def extract_forward_params(reference_code: str) -> tuple[str, str]:
    """Extract forward() parameters from reference code.

    Returns (full_params, param_names_only):
        full_params: "A: torch.Tensor, B: torch.Tensor"
        param_names_only: "A, B"
    """
    match = re.search(r"def forward\(self,\s*(.*?)\)\s*(?:->|:)", reference_code, re.DOTALL)
    if not match:
        return "*args", "*args"

    full_params = match.group(1).strip()

    # Extract just names
    names = []
    for param in full_params.split(","):
        param = param.strip()
        name = param.split(":")[0].strip()
        if name:
            names.append(name)

    return full_params, ", ".join(names)


def extract_model_init(reference_code: str) -> tuple[str, list[str]]:
    """Extract ``Model.__init__`` signature + arg names from the reference.

    Returns ``(full_signature_sans_self, arg_names)``::

        extract_model_init("class Model:\\n  def __init__(self, alpha=1.0): ...")
        # -> ("alpha=1.0", ["alpha"])

    Returns ``("", [])`` when Model has no ``__init__`` or the ref doesn't
    parse — the wrapper then falls back to a no-arg ctor (safe for simple
    problems like matmul). Uses ``ast`` rather than regex so the signature
    captures defaults, keyword args, and type annotations correctly.
    """
    if not reference_code or "class Model" not in reference_code:
        return "", []
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return "", []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "Model"):
            continue
        for body_node in node.body:
            if not (isinstance(body_node, ast.FunctionDef) and body_node.name == "__init__"):
                continue
            args = body_node.args
            pos = args.args[1:] if args.args and args.args[0].arg == "self" else args.args
            defaults = list(args.defaults)
            # Pad defaults so the last N args line up with the last N defaults.
            pad = [None] * (len(pos) - len(defaults))
            defaults = pad + defaults

            sig_pieces: list[str] = []
            names: list[str] = []
            for arg, default in zip(pos, defaults):
                names.append(arg.arg)
                piece = arg.arg
                if arg.annotation is not None:
                    try:
                        piece += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                if default is not None:
                    try:
                        piece += f" = {ast.unparse(default)}"
                    except Exception:
                        pass
                sig_pieces.append(piece)
            return ", ".join(sig_pieces), names
    return "", []


def _build_init_body(arg_names: list[str]) -> str:
    """Return the body lines for ``ModelNew.__init__`` as an indented string.

    Stores every positional argument as ``self.{name} = {name}`` so
    ``forward`` can route it into ``kernel_function``. Whether the forward
    actually USES each attribute is up to ``kernel_function`` — but storing
    them is zero-cost and matches what most ``Model.__init__`` bodies do.
    """
    if not arg_names:
        return "        pass"
    return "\n".join(f"        self.{n} = {n}" for n in arg_names)


def wrap_in_model_new(kernel_code: str, reference_code: str) -> str:
    """Wrap KernelAgent's kernel_function output in a ``ModelNew`` class
    whose ``__init__`` signature mirrors ``Model.__init__``.

    The harness calls ``ModelNew(*get_init_inputs())``, so a hardcoded
    no-arg ctor crashes with "takes 1 positional argument but N were
    given" for every parameterized KernelBench problem (Conv2D, Linear,
    GroupNorm, …). Mirroring the signature — with args stored as
    ``self.{name}`` and forwarded into ``kernel_function`` — fixes the
    whole class of problems at once without needing per-problem wiring.

    If the kernel already emits its own ``class ModelNew``, return as-is.
    """
    # Already has ModelNew — trust the LLM's version.
    if "class ModelNew" in kernel_code:
        return kernel_code

    full_params, param_names = extract_forward_params(reference_code)
    init_sig, init_args = extract_model_init(reference_code)

    # __init__ signature: either mirror Model's or fall back to no-arg.
    init_signature = f"self, {init_sig}" if init_sig else "self"
    init_body = _build_init_body(init_args)

    # Forward call: include any stored init args so kernel_function can
    # use them (e.g. ELU's alpha). kernel_function MUST accept the forward
    # tensors first, then the init attrs as keyword/positional — we use
    # keyword so signature mismatches become named-arg errors rather
    # than silent misalignment.
    kw_forward = ", ".join(f"{n}=self.{n}" for n in init_args)
    forward_call = f"kernel_function({param_names}"
    if kw_forward:
        forward_call += f", {kw_forward}"
    forward_call += ")"

    has_kernel_function = "def kernel_function" in kernel_code or "kernel_function" in kernel_code

    if has_kernel_function:
        return f'''import torch
import torch.nn as nn

{kernel_code}

class ModelNew(nn.Module):
    def __init__({init_signature}):
        super(ModelNew, self).__init__()
{init_body}

    def forward(self, {full_params}) -> torch.Tensor:
        return {forward_call}
'''
    # Kernel has @triton.jit but no kernel_function wrapper —
    # let the caller handle it.
    return f'''import torch
import torch.nn as nn

{kernel_code}
'''
