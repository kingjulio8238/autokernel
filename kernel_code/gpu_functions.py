"""Canonical GPU-to-Modal-function mapping.

Single source of truth for which Modal function handles each GPU type.
Imported by kernel_agent_bridge.py and kernel_agent/worker.py.
The copy in modal_infra/app.py is intentionally separate since Modal
containers cannot import from kernel_code.
"""

GPU_FUNCTION_MAP: dict[str, str] = {
    "L40S": "eval_kernel_on_gpu",
    "H100": "eval_kernel_h100",
    "A100-80GB": "eval_kernel_a100_80gb",
    "A100-40GB": "eval_kernel_a100_40gb",
}
