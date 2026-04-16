"""Integration layer -- bridges kernel-code TUI to the openkernel engine."""

from kernel_code.integration.openkernel_bridge import OpenKernelBridge
from kernel_code.integration.trace_bridge import TraceBridge

__all__ = ["OpenKernelBridge", "TraceBridge"]
