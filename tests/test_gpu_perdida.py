"""Un error fatal de GPU no se puede confundir con falta de memoria.

El fallo que esto fija: upscale() envolvia el pase de frame completo en un
`except Exception` desnudo y daba por supuesto que cualquier error era falta de
memoria, asi que se ponia a trocear. Pero cuando el driver de la GPU se
reinicia (TDR de Windows) el contexto CUDA queda roto para siempre: trocear no
arregla nada, cada bloque falla igual, y el mensaje que ve el usuario miente
sobre la causa.

Sacado de un caso real, el log del 16/11/2026:

    22:22:52  nvlddmkm: Resetting TDR occurred on GPUID:800
    22:35:06  nvlddmkm: Reset TDR occurred on GPUID:800
    22:35:09  "Frame completo no cabe en la GPU (CUDA failure 999...).
               Se continua por bloques de 256 px."

Las dos primeras lineas son del Visor de eventos de Windows; la tercera, del
programa diagnosticando mal lo que acababa de pasar.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhs_upscaler.inference import (
    GpuPerdida,
    TiledUpscaler,
    clasifica_error_gpu,
)

# El texto exacto que solto onnxruntime aquella noche.
ERROR_REAL = (
    "[ONNXRuntimeError] : 1 : FAIL : CUDA failure 999: unknown error ; GPU=0 ; "
    "hostname=MKDSK ; file=onnxruntime/core/providers/cuda/gpu_data_transfer.cc ; "
    "line=92 ; expr=cudaMemcpyAsync(dst_data, src_data, bytes, "
    "cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream.GetHandle()));"
)


class _Falso(TiledUpscaler):
    """Upscaler de mentira: falla donde se le diga y escala x2 con numpy."""

    scale = 2

    def __init__(self, revienta_con=None, revienta_en_tile=None):
        self.tile_size = 0
        self.overlap = 8
        self.fixed_tile = None
        self.revienta_con = revienta_con
        self.revienta_en_tile = revienta_en_tile
        self.llamadas = 0
        self.tiles = 0
        self.avisos: list[str] = []
        self.log = self.avisos.append

    def _run_tile(self, tile):
        # revienta_con falla solo en el primer intento: el del frame completo.
        self.llamadas += 1
        if self.revienta_con is not None and self.llamadas == 1:
            raise RuntimeError(self.revienta_con)
        self.tiles += 1
        if self.revienta_en_tile is not None and self.tiles == 2:
            raise RuntimeError(self.revienta_en_tile)
        return np.repeat(np.repeat(tile, self.scale, axis=0), self.scale, axis=1)


def frame(h=128, w=128):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ------------------------------------------------------ el clasificador ----

@pytest.mark.parametrize("texto", [
    ERROR_REAL,
    "CUDA failure 999: unknown error",
    "CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR",
    "an illegal memory access was encountered",
    "device-side assert triggered",
    "VK_ERROR_DEVICE_LOST",
])
def test_errores_fatales(texto):
    assert clasifica_error_gpu(RuntimeError(texto)) == "perdida"


@pytest.mark.parametrize("texto", [
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation",
    "Failed to allocate memory for requested buffer of size 1234",
    "std::bad_alloc",
    "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    "out of memory",
])
def test_errores_de_memoria(texto):
    assert clasifica_error_gpu(RuntimeError(texto)) == "memoria"


def test_lo_que_no_se_reconoce_no_se_declara_fatal():
    """Ante la duda, mejor intentar trocear que abortar la exportacion."""
    assert clasifica_error_gpu(RuntimeError(
        "Non-zero status code returned while running Conv node")) == "otro"


# --------------------------------------------------- el frame completo -----

def test_la_gpu_muerta_aborta_en_vez_de_trocear():
    """Lo que fallaba: esto se ponia a trocear y mentia sobre la causa."""
    up = _Falso(revienta_con=ERROR_REAL)
    with pytest.raises(GpuPerdida) as e:
        up.upscale(frame())

    assert up.tiles == 0, "se puso a trocear con el contexto CUDA ya roto"
    assert up.tile_size <= 0, "cambio a modo tiles para nada"
    assert not any("no cabe en la GPU" in a for a in up.avisos), \
        "sigue culpando a la memoria de un driver reiniciado"

    msg = str(e.value)
    assert "TDR" in msg
    assert "nvlddmkm" in msg, "no dice donde confirmarlo"
    assert "reanuda" in msg or "seguira por el segmento" in msg, \
        "no dice que el trabajo se puede reanudar"
    assert "999" in msg, "se perdio el error original"


def test_la_falta_de_memoria_si_trocea():
    up = _Falso(revienta_con="CUDA_ERROR_OUT_OF_MEMORY")
    salida = up.upscale(frame())
    assert salida.shape == (256, 256, 3)
    assert up.tile_size == 256
    assert any("no cabe en la GPU" in a for a in up.avisos)


def test_un_error_desconocido_trocea_pero_no_inventa_la_causa():
    up = _Falso(revienta_con="Conv node failed for reasons unknown to man")
    salida = up.upscale(frame())
    assert salida.shape == (256, 256, 3)
    assert any("no se pudo procesar de una vez" in a for a in up.avisos), \
        "afirma que no cabe en la GPU sin saberlo"
    assert not any("no cabe en la GPU" in a for a in up.avisos)


# ---------------------------------------------------------- los tiles ------

def test_la_gpu_muerta_a_mitad_del_troceado_tambien_se_entiende():
    """El TDR puede llegar cuando ya se va por bloques."""
    up = _Falso(revienta_en_tile=ERROR_REAL)
    up.tile_size = 64
    with pytest.raises(GpuPerdida) as e:
        up.upscale(frame())
    assert "TDR" in str(e.value)


def test_un_error_normal_en_un_tile_se_propaga_tal_cual():
    """Solo se reescribe lo que de verdad es la GPU cayendose."""
    up = _Falso(revienta_en_tile="Conv node se atraganto")
    up.tile_size = 64
    with pytest.raises(RuntimeError) as e:
        up.upscale(frame())
    assert not isinstance(e.value, GpuPerdida)
    assert "atraganto" in str(e.value)


def test_el_troceado_normal_sigue_dando_el_tamano_correcto():
    up = _Falso()
    up.tile_size = 64
    assert up.upscale(frame(96, 128)).shape == (192, 256, 3)


# ------------------------------------------------- el aviso del proveedor --

def test_el_aviso_no_da_un_solo_consejo():
    """Decir solo "ejecuta sync.bat" es el consejo equivocado tras un TDR."""
    import inspect

    from vhs_upscaler import inference

    fuente = inspect.getsource(inference.OnnxUpscaler.__init__)
    assert "TDR" in fuente, "el aviso no contempla que la GPU se haya reiniciado"
    assert "sync.bat" in fuente, "se perdio el consejo para el runtime que falta"
