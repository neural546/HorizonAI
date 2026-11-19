"""Detect hardware resources and select an appropriate profile."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceInfo:
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    vram_gb: float


def _get_ram_gb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


def detect_resources() -> ResourceInfo:
    cpu = os.cpu_count() or 1
    ram = _get_ram_gb()
    cuda = False
    vram = 0.0
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        cuda = False
    return ResourceInfo(cpu_count=cpu, ram_gb=ram, cuda_available=cuda, vram_gb=vram)


def detect_profile(info: Optional[ResourceInfo] = None) -> str:
    info = info or detect_resources()
    if info.cuda_available:
        if info.vram_gb >= 15.0:
            return "high"
        if info.vram_gb >= 7.0:
            return "medium"
        return "low"
    if info.ram_gb >= 32.0 and info.cpu_count >= 8:
        return "medium"
    return "low"


def report() -> str:
    info = detect_resources()
    return "\n".join([
        "Resource Report",
        "===============",
        f"CPU count:     {info.cpu_count}",
        f"RAM (GB):      {info.ram_gb:.1f}",
        f"CUDA available:{info.cuda_available}",
        f"VRAM (GB):     {info.vram_gb:.1f}",
        f"Suggested profile: {detect_profile(info)}",
    ])


if __name__ == "__main__":
    print(report())
    sys.exit(0)
