"""Deteccion de hardware y resolucion del dispositivo de inferencia.

Reglas:
- La CPU siempre esta disponible como respaldo.
- CUDA solo si onnxruntime-gpu esta instalado Y hay GPU NVIDIA.
- DirectML solo si onnxruntime-directml esta instalado (Windows).
- Si el dispositivo pedido no esta disponible, se degrada a CPU con aviso.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

DEVICE_LABELS = {
    "auto": "Automatico (recomendado)",
    "cpu": "CPU (respaldo, siempre disponible)",
    "cuda": "GPU NVIDIA (CUDA)",
    "directml": "GPU integrada / DirectML (AMD, Intel)",
    "vulkan": "GPU via Vulkan / ncnn (rapido en AMD)",
}


@dataclass
class HardwareInfo:
    nvidia_gpu: str | None = None
    available_providers: list[str] = field(default_factory=list)
    vulkan_gpus: int = 0

    @property
    def cuda_ready(self) -> bool:
        return "CUDAExecutionProvider" in self.available_providers

    @property
    def directml_ready(self) -> bool:
        return "DmlExecutionProvider" in self.available_providers

    @property
    def vulkan_ready(self) -> bool:
        return self.vulkan_gpus > 0

    def device_available(self, device: str) -> bool:
        if device in ("auto", "cpu"):
            return True
        if device == "cuda":
            return self.cuda_ready
        if device == "directml":
            return self.directml_ready
        if device == "vulkan":
            return self.vulkan_ready
        return False


def detect_nvidia_gpu() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def get_vulkan_gpu_count() -> int:
    """Cuantas GPUs ve ncnn por Vulkan. Cero si ncnn no esta o falla.

    get_gpu_count() crea la instancia Vulkan de forma implicita. Si se queda
    viva, al cerrar el interprete la liberacion llega en mal orden y el proceso
    muere con SIGSEGV (codigo 139) DESPUES de haber hecho su trabajo: el video
    sale bien, pero quien mire el codigo de salida cree que fallo.

    Aqui solo hace falta contar, asi que se destruye enseguida. Si luego se usa
    de verdad el backend Vulkan, ncnn la vuelve a crear sola (comprobado:
    contar, destruir y volver a contar funciona), y de cerrarla se encarga
    NcnnUpscaler.close().
    """
    try:
        import ncnn
    except Exception:
        return 0
    try:
        return int(ncnn.get_gpu_count())
    except Exception:
        return 0
    finally:
        close_vulkan(ncnn)


def close_vulkan(ncnn_module=None) -> None:
    """Libera la instancia Vulkan de ncnn. Se puede llamar de mas: es idempotente."""
    try:
        if ncnn_module is None:
            import ncnn as ncnn_module
        ncnn_module.destroy_gpu_instance()
    except Exception:
        pass


def get_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        # ImportError si falta el paquete; OSError/otros si falta una DLL del runtime.
        return []


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        nvidia_gpu=detect_nvidia_gpu(),
        available_providers=get_ort_providers(),
        vulkan_gpus=get_vulkan_gpu_count(),
    )


def resolve_device(requested: str, hw: HardwareInfo) -> tuple[str, str | None]:
    """Devuelve (dispositivo_real, aviso_o_none). Nunca falla: cae a CPU."""
    if requested == "auto":
        if hw.cuda_ready:
            return "cuda", None
        if hw.vulkan_ready:
            return "vulkan", None
        if hw.directml_ready:
            return "directml", None
        return "cpu", None
    if hw.device_available(requested):
        return requested, None
    aviso = (
        f"El dispositivo '{requested}' no esta disponible en esta PC "
        f"(falta el runtime o el hardware). Se usara CPU como respaldo."
    )
    return "cpu", aviso
