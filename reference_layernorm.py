from utils import make_match_reference, DeterministicContext
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of LayerNorm using PyTorch.

    Applies layer normalization along the last dimension:
        y = (x - mean) / sqrt(var + eps) * weight + bias

    This is a realistic memory-bound kernel with ~16MB of tensor data —
    large enough that kernel launch overhead is amortized, giving
    Triton real room to optimize via fused single-pass reductions
    and vectorized loads.
    """
    with DeterministicContext():
        x, weight, bias, output = data
        # Layer normalize along the last dimension
        normalized = torch.nn.functional.layer_norm(
            x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5
        )
        output[...] = normalized
        return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensors for LayerNorm.

    Shape: (batch=size, hidden=size) with fp16.
    At size=2048 -> 2048 * 2048 * 2B = 8MB input tensor.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    # Use size=2048 for realistic workload: 4M elements, ~8MB per tensor
    batch = size * 16  # e.g. 2048 if size=128
    hidden = size * 16
    x = torch.randn(
        batch, hidden, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    weight = torch.randn(
        hidden, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    bias = torch.randn(
        hidden, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    output = torch.empty(batch, hidden, device="cuda", dtype=torch.float16).contiguous()
    return x, weight, bias, output


check_implementation = make_match_reference(ref_kernel)
