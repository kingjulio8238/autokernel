from utils import make_match_reference, DeterministicContext
import torch
from task import input_t, output_t


WORKLOAD_SPEC = {
    "size": 4_194_304,
    "seed": 42,
    "contention": 90,
}


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of 256-bin histogram over uint8 data.

    Counts occurrences of each byte value into a fixed 256-bin output.
    Atomic contention is the hard case: when most inputs land in the
    same bin, naive atomicAdd on global memory serializes. Real upside
    comes from per-block shared-memory histograms + warp-aggregated
    stores, or from stochastic/privatized bins.
    """
    with DeterministicContext():
        data, output = data
        output[...] = torch.bincount(data, minlength=256)
        return output


def generate_input(size: int = 30080, seed: int = 0, contention: int = 90) -> input_t:
    """
    Generates an input tensor and a reusable output buffer.

    size      : number of uint8 elements.
    seed      : RNG seed.
    contention: percentage [0..100] of entries forced to a single value.
                Default 90 crushes naive atomicAdd kernels. Lower = easier.

    WORKLOAD_SPEC (module-level) declares the benchmark shape used by the
    Modal harness; it is passed in as kwargs.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    data = torch.randint(
        0, 256, (size,), device="cuda", dtype=torch.uint8, generator=gen
    )

    # Force a fraction of entries to a single value — atomic hot-spot.
    evil_value = torch.randint(
        0, 256, (), device="cuda", dtype=torch.uint8, generator=gen
    )
    evil_loc = torch.rand(
        (size,), device="cuda", dtype=torch.float32, generator=gen
    ) < (contention / 100.0)
    data[evil_loc] = evil_value

    output = torch.empty(256, device="cuda", dtype=torch.int64).contiguous()
    return data.contiguous(), output


check_implementation = make_match_reference(ref_kernel)
