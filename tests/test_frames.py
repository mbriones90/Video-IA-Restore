"""Lectura de frames crudos: el fallo que dio la imagen repetida 9 veces.

vspipe escribe RGB24 en PLANOS separados y en orden GBR (el GBRP de FFmpeg),
no entrelazado. El pipeline lo leia como BGR packed, y cada plano acababa
repartido en una rejilla 3x3 de copias aplastadas del fotograma.

Esta prueba fija el contrato: dado un buffer planar GBR, el frame que sale
tiene que ser BGR entrelazado con los canales en su sitio.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhs_upscaler.pipeline import Deflicker


def planar_gbr_to_bgr(raw: bytes, height: int, width: int) -> np.ndarray:
    """Misma conversion que hace el lector del pipeline (ver _pump)."""
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(3, height, width)
    return np.stack((planes[1], planes[0], planes[2]), axis=-1)


def test_planar_gbr_se_convierte_a_bgr():
    h, w = 4, 5
    r = np.full((h, w), 10, np.uint8)
    g = np.full((h, w), 20, np.uint8)
    b = np.full((h, w), 30, np.uint8)
    # Orden en el que vspipe vuelca los planos: G, B, R.
    raw = g.tobytes() + b.tobytes() + r.tobytes()

    frame = planar_gbr_to_bgr(raw, h, w)

    assert frame.shape == (h, w, 3)
    assert np.all(frame[:, :, 0] == 30), "canal B incorrecto"
    assert np.all(frame[:, :, 1] == 20), "canal G incorrecto"
    assert np.all(frame[:, :, 2] == 10), "canal R incorrecto"


def test_leer_planar_como_packed_produce_la_rejilla_3x3():
    """Reproduce el bug original, para que quede claro que esto era aquello."""
    h, w = 12, 12
    # Un degradado vertical: si se lee mal, deja de ser monotono por filas.
    plane = np.tile(np.arange(h, dtype=np.uint8)[:, None] * 20, (1, w))
    raw = plane.tobytes() * 3

    malo = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)   # lectura ERRONEA
    bueno = planar_gbr_to_bgr(raw, h, w)                          # lectura correcta

    # En la lectura correcta cada fila es constante (es un degradado vertical).
    assert all(len(np.unique(bueno[y, :, 0])) == 1 for y in range(h))
    # En la erronea no lo es: ahi se ve la rejilla.
    assert not all(len(np.unique(malo[y, :, 0])) == 1 for y in range(h))


def test_frame_bgr_sobrevive_ida_y_vuelta():
    rng = np.random.default_rng(0)
    h, w = 8, 6
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    raw = (frame[:, :, 1].tobytes()   # G
           + frame[:, :, 0].tobytes()  # B
           + frame[:, :, 2].tobytes())  # R
    assert np.array_equal(planar_gbr_to_bgr(raw, h, w), frame)


# --------------------------------------------------------------- deflicker --

def test_deflicker_acerca_el_brillo_a_la_media():
    d = Deflicker(window=5)
    base = np.full((10, 10, 3), 100, np.uint8)
    for _ in range(4):
        d.apply(base)
    # Un fotograma mas oscuro de lo normal debe subir hacia la media.
    oscuro = np.full((10, 10, 3), 92, np.uint8)
    salida = d.apply(oscuro)
    assert salida.mean() > oscuro.mean()
    assert salida.mean() <= base.mean() + 1


def test_deflicker_no_toca_un_fundido_a_negro():
    d = Deflicker(window=5)
    for _ in range(4):
        d.apply(np.full((8, 8, 3), 120, np.uint8))
    negro = np.zeros((8, 8, 3), np.uint8)
    assert np.array_equal(d.apply(negro), negro)


def test_deflicker_no_amplifica_un_corte_de_plano():
    """Un cambio de escena real no es parpadeo: la correccion va topada."""
    d = Deflicker(window=5)
    for _ in range(4):
        d.apply(np.full((8, 8, 3), 200, np.uint8))
    corte = np.full((8, 8, 3), 40, np.uint8)
    salida = d.apply(corte)
    # Como mucho un 12 % de ganancia, no un salto de 40 a 200.
    assert salida.mean() == pytest.approx(40 * 1.12, abs=1.5)


def test_deflicker_primer_frame_pasa_tal_cual():
    d = Deflicker()
    f = np.full((4, 4, 3), 77, np.uint8)
    assert np.array_equal(d.apply(f), f)
