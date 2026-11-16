"""Pausa y reanudacion del procesamiento."""

from __future__ import annotations

import threading
import time

import pytest

from vhs_upscaler.pipeline import PauseToken, ProcessingCancelled


# ------------------------------------------------------------- estado ------

def test_arranca_corriendo():
    assert PauseToken().is_paused is False


def test_pausar_y_reanudar():
    t = PauseToken()
    t.pause()
    assert t.is_paused
    t.resume()
    assert not t.is_paused


def test_toggle_devuelve_el_estado_nuevo():
    t = PauseToken()
    assert t.toggle() is True    # queda en pausa
    assert t.toggle() is False   # vuelve a correr


def test_pausar_dos_veces_no_duplica_la_cuenta():
    t = PauseToken()
    t.pause()
    time.sleep(0.15)
    t.pause()          # ya estaba en pausa: no debe reiniciar el cronometro
    time.sleep(0.05)
    t.resume()
    assert t.paused_seconds == pytest.approx(0.2, abs=0.12)


def test_reanudar_sin_estar_en_pausa_no_hace_nada():
    t = PauseToken()
    t.resume()
    assert not t.is_paused
    assert t.paused_seconds == 0.0


# ------------------------------------------------------------- espera ------

def test_wait_no_cuesta_nada_cuando_corre():
    t = PauseToken()
    inicio = time.monotonic()
    for _ in range(1000):
        t.wait()
    assert time.monotonic() - inicio < 0.1, "el camino rapido se perdio"


def test_wait_bloquea_mientras_esta_en_pausa():
    t = PauseToken()
    t.pause()
    soltado = threading.Event()

    def trabajador():
        t.wait()
        soltado.set()

    hilo = threading.Thread(target=trabajador, daemon=True)
    hilo.start()
    assert not soltado.wait(0.3), "no deberia haber pasado todavia"
    t.resume()
    assert soltado.wait(2.0), "no se solto al reanudar"


def test_se_puede_cancelar_estando_en_pausa():
    """Si no, cerrar la app con el proceso pausado se quedaria colgado."""
    t = PauseToken()
    cancel = threading.Event()
    t.pause()
    fallo: list[BaseException] = []

    def trabajador():
        try:
            t.wait(cancel)
        except BaseException as exc:  # noqa: BLE001
            fallo.append(exc)

    hilo = threading.Thread(target=trabajador, daemon=True)
    hilo.start()
    time.sleep(0.1)
    cancel.set()
    hilo.join(timeout=2.0)
    assert not hilo.is_alive()
    assert fallo and isinstance(fallo[0], ProcessingCancelled)


# -------------------------------------------------- tiempo acumulado -------

def test_el_tiempo_en_pausa_se_acumula():
    t = PauseToken()
    for _ in range(2):
        t.pause()
        time.sleep(0.12)
        t.resume()
        time.sleep(0.02)
    assert t.paused_seconds == pytest.approx(0.24, abs=0.12)


def test_el_tiempo_en_pausa_incluye_la_pausa_en_curso():
    t = PauseToken()
    t.pause()
    time.sleep(0.15)
    assert t.paused_seconds >= 0.1, "una pausa abierta tambien cuenta"


def test_sin_pausas_el_tiempo_detenido_es_cero():
    t = PauseToken()
    t.wait()
    assert t.paused_seconds == 0.0


# ------------------------------------------------------- concurrencia ------

def test_aguanta_pausas_y_reanudaciones_desde_varios_hilos():
    t = PauseToken()
    errores: list[BaseException] = []

    def machacar():
        try:
            for _ in range(200):
                t.toggle()
        except BaseException as exc:  # noqa: BLE001
            errores.append(exc)

    hilos = [threading.Thread(target=machacar) for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=5)
    assert not errores
    t.resume()
    assert not t.is_paused
    assert t.paused_seconds >= 0.0
