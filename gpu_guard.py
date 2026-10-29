"""Keep every run strictly inside the GPU's own memory.

On Windows the NVIDIA driver silently spills CUDA allocations that do not fit in VRAM into shared system memory
("sysmem fallback").  On this workstation (VBS/HVCI enabled) every run whose peak demand exceeded the 32 GB of the
RTX 5090 ended in bugcheck 0x20001 HYPERVISOR_ERROR within minutes.  Capping PyTorch's allocator below the VRAM size
turns that situation into an ordinary CUDA out-of-memory error instead of a machine crash.

Import this module before any CUDA work.  The cap can be changed with the environment variable GPU_MEM_FRACTION.
"""
import os

import torch

FRACTION = float(os.environ.get("GPU_MEM_FRACTION", "0.85"))

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(FRACTION, 0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[gpu_guard] allocator capped at {FRACTION:.2f} x {total:.1f} GiB = {FRACTION * total:.1f} GiB", flush=True)
