"""El proceso tiene que terminar con codigo 0, no con SIGSEGV.

El fallo que esto fija: ncnn.get_gpu_count() crea la instancia Vulkan de forma
implicita y nadie la destruia. Al cerrar el interprete, la liberacion llegaba
en mal orden y el proceso moria con SIGSEGV (codigo 139) DESPUES de haber
terminado su trabajo y escrito el archivo.

Lo peor no era el ruido: el video salia bien, pero cualquier .bat o script que
mirara el codigo de salida creia que habia fallado, y un error de verdad
habria quedado tapado por el mismo 139.

Reducido a esto:
    import ncnn; ncnn.get_gpu_count()                         -> codigo 139
    import ncnn; ncnn.get_gpu_count(); destroy_gpu_instance()  -> codigo 0

Un segfault no se puede comprobar dentro del propio proceso: estos tests
lanzan un subproceso y miran con que codigo vuelve.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from vhs_upscaler.hardware import close_vulkan, get_vulkan_gpu_count


def correr(codigo: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(codigo)],
        capture_output=True, timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# ------------------------------------------------- sin tocar la GPU -------

class _NcnnFalso:
    """ncnn de mentira, para comprobar la secuencia sin depender de Vulkan."""

    def __init__(self, gpus=1, revienta_al_contar=False):
        self.gpus = gpus
        self.revienta_al_contar = revienta_al_contar
        self.destruido = 0

    def get_gpu_count(self):
        if self.revienta_al_contar:
            raise RuntimeError("sin driver Vulkan")
        return self.gpus

    def destroy_gpu_instance(self):
        self.destruido += 1


def test_contar_gpus_cierra_siempre_la_instancia(monkeypatch):
    falso = _NcnnFalso(gpus=2)
    monkeypatch.setitem(sys.modules, "ncnn", falso)
    assert get_vulkan_gpu_count() == 2
    assert falso.destruido == 1, "la instancia Vulkan se quedo viva"


def test_la_cierra_incluso_si_contar_falla(monkeypatch):
    """Si get_gpu_count revienta a medias, puede haberla creado ya."""
    falso = _NcnnFalso(revienta_al_contar=True)
    monkeypatch.setitem(sys.modules, "ncnn", falso)
    assert get_vulkan_gpu_count() == 0
    assert falso.destruido == 1


def test_sin_ncnn_instalado_no_estalla(monkeypatch):
    def no_importable(nombre, *a, **k):
        if nombre == "ncnn":
            raise ImportError("no esta")
        return original(nombre, *a, **k)

    import builtins
    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", no_importable)
    assert get_vulkan_gpu_count() == 0


def test_cerrar_de_mas_es_inofensivo(monkeypatch):
    """close() y el atexit se solapan: cerrar dos veces no puede fallar."""
    falso = _NcnnFalso()
    close_vulkan(falso)
    close_vulkan(falso)
    assert falso.destruido == 2

    class Roto:
        def destroy_gpu_instance(self):
            raise RuntimeError("ya estaba cerrada")

    close_vulkan(Roto())  # no debe propagar


# ------------------------------------------------- de verdad, con GPU ----

@pytest.mark.slow
def test_contar_gpus_no_mata_al_proceso():
    """El caso reducido: antes devolvia 139."""
    pytest.importorskip("ncnn")
    p = correr("""
        from vhs_upscaler.hardware import get_vulkan_gpu_count
        print(get_vulkan_gpu_count())
    """)
    assert p.returncode == 0, (
        f"codigo {p.returncode} "
        f"({'SIGSEGV' if p.returncode == 139 else 'otro'}); "
        f"{p.stderr.decode('utf-8', 'replace')[-300:]}")


@pytest.mark.slow
def test_detectar_hardware_no_mata_al_proceso():
    pytest.importorskip("ncnn")
    p = correr("""
        from vhs_upscaler.hardware import detect_hardware
        hw = detect_hardware()
        print(hw.vulkan_gpus, hw.cuda_ready)
    """)
    assert p.returncode == 0, f"codigo {p.returncode}"


@pytest.mark.slow
def test_el_upscaler_ncnn_se_cierra_solo():
    """Aunque nadie llame a close(), el atexit tiene que salvar la salida."""
    pytest.importorskip("ncnn")
    p = correr("""
        from vhs_upscaler.hardware import detect_hardware
        from vhs_upscaler.inference import create_upscaler

        if detect_hardware().vulkan_gpus:
            up = create_upscaler("realesr-general-x4v3", "vulkan", log=lambda *_: None)
            print("creado", type(up).__name__)
        else:
            print("sin vulkan")
    """)
    assert p.returncode == 0, (
        f"codigo {p.returncode}; {p.stderr.decode('utf-8', 'replace')[-300:]}")
