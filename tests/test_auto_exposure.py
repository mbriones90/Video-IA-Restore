"""Exposicion automatica por escena.

El contrato que fijan estas pruebas, medido sobre una cinta familiar real:
una escena oscura y una clara tienen el mismo punto de negro y el mismo de
blanco, y solo difieren en la mediana. Asi que la correccion tiene que mover
los medios tonos SIN tocar los extremos.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vhs_upscaler.pipeline import AutoExposure, ProcessOptions, build_auto_exposure


def escena(mediana: int, negro: int = 14, blanco: int = 229,
           size: tuple[int, int] = (120, 160)) -> np.ndarray:
    """Fotograma sintetico con negro y blanco fijos y la mediana pedida.

    Mitad de los pixeles por debajo de la mediana y mitad por encima, mas unos
    pocos en los extremos para que los percentiles 1 y 99 caigan donde se dice.
    """
    h, w = size
    total = h * w
    bajos = np.linspace(negro, mediana, total // 2)
    altos = np.linspace(mediana, blanco, total - total // 2)
    gris = np.concatenate([bajos, altos]).astype(np.uint8)
    rng = np.random.default_rng(0)
    rng.shuffle(gris)
    return np.repeat(gris.reshape(h, w, 1), 3, axis=2)


def mediana(frame: np.ndarray) -> float:
    return float(np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def extremos(frame: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, p99 = np.percentile(g, (1.0, 99.0))
    return float(p1), float(p99)


# ------------------------------------------------------ comportamiento base --

def test_una_escena_oscura_se_aclara_hacia_el_objetivo():
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    oscura = escena(28)
    salida = ae.apply(oscura)
    assert mediana(salida) > mediana(oscura)
    assert mediana(salida) == pytest.approx(56, abs=8)


def test_una_escena_ya_correcta_casi_no_se_toca():
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    salida = ae.apply(escena(56))
    assert mediana(salida) == pytest.approx(56, abs=4)


def test_una_escena_demasiado_clara_se_oscurece_solo_un_poco():
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    clara = escena(140)
    salida = ae.apply(clara)
    assert mediana(salida) < mediana(clara)


def test_la_correccion_es_asimetrica():
    """Aclarar recupera caras; oscurecer hunde el fondo a negro.

    Probado sobre la cinta real: bajar una escena diurna al mismo nivel que una
    nocturna convertia la ropa oscura del fondo en un borron. Asi que el
    objetivo funciona casi como suelo.
    """
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    subida = mediana(ae.apply(escena(28))) - 28       # escena oscura
    bajada = 140 - mediana(AutoExposure(target=56, strength=100,
                                        smoothing=1).apply(escena(140)))
    assert subida > bajada * 1.5, "deberia aclarar mucho mas de lo que oscurece"


def test_una_escena_muy_clara_sigue_pareciendo_clara():
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    salida = ae.apply(escena(200))
    assert mediana(salida) > 170, "una escena soleada no debe quedar apagada"


def test_oscurecer_no_aplasta_las_sombras():
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    clara = escena(140, negro=10)
    p1_antes, _ = extremos(clara)
    p1_despues, _ = extremos(ae.apply(clara))
    assert p1_despues > p1_antes - 6, "las sombras se hundieron contra el negro"


def test_el_negro_y_el_blanco_no_se_mueven():
    """Lo que distingue esto de un estirado de niveles."""
    ae = AutoExposure(target=56, strength=100, smoothing=1)
    oscura = escena(28, negro=14, blanco=229)
    salida = ae.apply(oscura)
    p1, p99 = extremos(salida)
    assert p1 == pytest.approx(14, abs=4), "el negro se levanto"
    assert p99 == pytest.approx(229, abs=5), "las altas luces se quemaron"


def test_no_hay_recorte_en_las_altas_luces():
    ae = AutoExposure(target=90, strength=100, smoothing=1)
    salida = ae.apply(escena(28, blanco=229))
    quemados = (cv2.cvtColor(salida, cv2.COLOR_BGR2GRAY) >= 255).mean()
    assert quemados < 0.005


# ------------------------------------------------------------- intensidad ---

def test_intensidad_cero_no_cambia_nada():
    ae = AutoExposure(target=56, strength=0, smoothing=1)
    oscura = escena(28)
    assert mediana(ae.apply(oscura)) == pytest.approx(mediana(oscura), abs=1)


def test_la_intensidad_gradua_la_correccion():
    oscura = escena(28)
    suave = AutoExposure(target=56, strength=40, smoothing=1).apply(oscura)
    fuerte = AutoExposure(target=56, strength=100, smoothing=1).apply(oscura)
    assert mediana(oscura) < mediana(suave) < mediana(fuerte)


def test_la_correccion_esta_topada():
    """Una escena nocturna a proposito no debe quedar gris lavada."""
    ae = AutoExposure(target=160, strength=100, smoothing=1)
    inv = ae.measure(escena(20))
    assert inv >= ae.min_inv


# --------------------------------------------------------------- escenas ----

def test_un_corte_de_plano_reinicia_el_suavizado():
    ae = AutoExposure(target=56, strength=100, smoothing=30)
    for _ in range(10):
        ae.apply(escena(140))        # escena clara, llena la media movil
    assert len(ae._inv) > 1
    ae.apply(escena(28))             # corte a una escena oscura
    assert len(ae._inv) == 1, "no se reinicio: arrastraria la exposicion anterior"


def test_sin_corte_el_suavizado_acumula():
    ae = AutoExposure(target=56, strength=100, smoothing=30)
    base = escena(40)
    for _ in range(5):
        ae.apply(base)
    assert len(ae._inv) == 5


def test_el_suavizado_amortigua_un_salto_de_un_solo_frame():
    """Un fotograma raro no debe cambiar la exposicion de la escena."""
    ae = AutoExposure(target=56, strength=100, smoothing=30)
    base = escena(40)
    for _ in range(20):
        ae.apply(base)
    estable = ae.measure(base)
    # Mismo encuadre pero un poco mas oscuro: no es corte, es ruido de AGC.
    saltado = ae.measure(escena(34, size=(120, 160)))
    assert abs(saltado - estable) < 0.08


# ----------------------------------------------------------- casos limite ---

def test_un_fundido_a_negro_no_se_toca():
    ae = AutoExposure(target=56, strength=100, smoothing=5)
    negro = np.zeros((60, 80, 3), np.uint8)
    assert ae.measure(negro) is None
    assert np.array_equal(ae.apply(negro), negro)


def test_una_pantalla_plana_no_se_toca():
    ae = AutoExposure(target=56, strength=100, smoothing=5)
    plano = np.full((60, 80, 3), 120, np.uint8)
    assert ae.measure(plano) is None


def test_no_revienta_con_un_frame_minusculo():
    ae = AutoExposure(target=56, strength=100, smoothing=5)
    assert ae.apply(escena(30, size=(8, 8))).shape == (8, 8, 3)


# --------------------------------------------------------------- fabrica ----

def test_la_fabrica_traslada_las_opciones():
    opts = ProcessOptions(input_path="x", output_path="y",
                          auto_exposure_target=70, auto_exposure_strength=50,
                          auto_exposure_smoothing=12)
    ae = build_auto_exposure(opts)
    assert ae.target == 70
    assert ae.strength == pytest.approx(0.5)
    assert ae._inv.maxlen == 12
