"""
GPU/CPU device selection shared by all ML services.

Usage:
    from app.core.device import device_manager
    dev = device_manager.torch_device()    # "cuda:0" or "cpu"
    device_manager.is_gpu()                # bool

Set DEVICE=cpu|gpu|auto in .env (default: auto). GPU_ID picks which CUDA
device when multiple GPUs are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("core.device")


@dataclass(frozen=True)
class DeviceInfo:
    mode: str
    use_gpu: bool
    torch_device: str
    gpu_name: str | None
    gpu_id: int


class DeviceManager:
    """Single source of truth for GPU/CPU decisions."""

    def __init__(self) -> None:
        self._info: DeviceInfo | None = None

    def info(self) -> DeviceInfo:
        if self._info is None:
            self._info = self._resolve()
        return self._info

    def is_gpu(self) -> bool:
        return self.info().use_gpu

    def torch_device(self) -> str:
        return self.info().torch_device

    def describe(self) -> dict:
        info = self.info()
        return {
            "mode": info.mode,
            "useGpu": info.use_gpu,
            "torchDevice": info.torch_device,
            "gpuName": info.gpu_name,
            "gpuId": info.gpu_id,
        }

    def _resolve(self) -> DeviceInfo:
        mode = settings.device.mode.lower()
        gpu_id = settings.device.gpu_id
        if mode == "cpu":
            logger.info("Device: forced CPU (DEVICE=cpu)")
            return DeviceInfo(mode="cpu", use_gpu=False, torch_device="cpu", gpu_name=None, gpu_id=-1)

        cuda_available, gpu_name = _probe_cuda(gpu_id)
        if mode == "gpu":
            if not cuda_available:
                logger.warning("DEVICE=gpu requested but CUDA unavailable; falling back to CPU")
                return DeviceInfo(mode="gpu->cpu", use_gpu=False, torch_device="cpu", gpu_name=None, gpu_id=-1)
            _configure_torch_gpu()
            logger.info("Device: GPU cuda:%d (%s)", gpu_id, gpu_name)
            return DeviceInfo(mode="gpu", use_gpu=True, torch_device=f"cuda:{gpu_id}", gpu_name=gpu_name, gpu_id=gpu_id)

        # auto
        if cuda_available:
            _configure_torch_gpu()
            logger.info("Device: auto-selected GPU cuda:%d (%s)", gpu_id, gpu_name)
            return DeviceInfo(mode="auto", use_gpu=True, torch_device=f"cuda:{gpu_id}", gpu_name=gpu_name, gpu_id=gpu_id)
        logger.info("Device: auto-selected CPU (no CUDA)")
        return DeviceInfo(mode="auto", use_gpu=False, torch_device="cpu", gpu_name=None, gpu_id=-1)


@lru_cache(maxsize=1)
def _probe_cuda(gpu_id: int) -> tuple[bool, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        count = torch.cuda.device_count()
        if count <= 0:
            return False, None
        idx = max(0, min(gpu_id, count - 1))
        return True, torch.cuda.get_device_name(idx)
    except Exception as exc:
        logger.debug("CUDA probe failed: %s", exc)
        return False, None


def _configure_torch_gpu() -> None:
    try:
        import torch

        if settings.device.torch_allow_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:  # older torch
                pass
    except Exception:
        pass


device_manager = DeviceManager()
