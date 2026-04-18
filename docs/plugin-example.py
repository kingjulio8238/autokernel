"""Example kernel code plugin: adds a custom optimization skill."""

PLUGIN_SPEC = {
    "name": "custom_gemm_skill",
    "version": "0.1.0",
    "type": "skill",
    "description": "Custom GEMM optimization skill for H100 Tensor Cores",
    "author": "kernel-engineer",
}


def register(skill_library):
    """Register custom skills with the skill library."""
    skill_library.add_skill({
        "id": "custom_h100_gemm",
        "name": "H100 Tensor Core GEMM",
        "trigger": "GEMM on H100 with Hopper architecture",
        "approach": "Use TMA for async global->shared memory copy...",
        "backend": "cuda",
        "evidence": [],
        "code_template": None,
        "tags": ["gemm", "h100", "hopper", "tma"],
    })
