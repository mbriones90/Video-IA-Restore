"""Pruebas de extremo a extremo: exportar de verdad, troceado y reanudacion.

Usan el modelo bicubico (sin IA) para correr en segundos en cualquier PC, y
el clip sintetico de `conftest`. Lo que se comprueba es el pipeline, no la
calidad del escalado.
"""

from __future__ import annotations

import subprocess
import threading

import pytest

from vhs_upscaler.pipeline import (
    PauseToken, ProcessingCancelled, ProcessOptions, VideoProcessor,
    find_resumable_jobs,
)

pytestmark = pytest.mark.slow


def contar_frames(ffmpeg: str, path) -> tuple[int, float]:
    """Devuelve (frames, duracion) leyendo el archivo con OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        return frames, frames / fps
    finally:
        cap.release()


def tiene_audio(ffmpeg: str, path) -> bool:
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                          capture_output=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return b"Audio:" in proc.stderr


def base_opts(sample_video, tmp_path, **kw) -> ProcessOptions:
    cfg = dict(
        input_path=str(sample_video),
        output_path=str(tmp_path / "salida.mp4"),
        model_id="bicubic", device="cpu",
        deinterlace=False, denoise=False,
        target_width=1280, target_height=720,
        quality="rapida",
    )
    cfg.update(kw)
    return ProcessOptions(**cfg)


def test_export_simple(sample_video, tmp_path, ffmpeg, jobs_dir):
    opts = base_opts(sample_video, tmp_path, segment_seconds=0)
    out = VideoProcessor().run(opts, log=lambda *_: None)

    assert out.exists() and out.stat().st_size > 0
    frames, dur = contar_frames(ffmpeg, out)
    assert frames == pytest.approx(60, abs=2)   # 2 s a 30 fps
    assert dur == pytest.approx(2.0, abs=0.2)
    assert tiene_audio(ffmpeg, out), "el audio del original debe llegar a la salida"


def test_el_troceado_da_el_mismo_resultado_que_sin_trocear(
        sample_video, tmp_path, ffmpeg, jobs_dir):
    """Si los segmentos se solapan o se pisan, el conteo no cuadra."""
    entero = VideoProcessor().run(
        base_opts(sample_video, tmp_path, output_path=str(tmp_path / "entero.mp4"),
                  segment_seconds=0),
        log=lambda *_: None)
    troceado = VideoProcessor().run(
        base_opts(sample_video, tmp_path, output_path=str(tmp_path / "troceado.mp4"),
                  segment_seconds=1),
        log=lambda *_: None)

    f1, d1 = contar_frames(ffmpeg, entero)
    f2, d2 = contar_frames(ffmpeg, troceado)
    assert f2 == pytest.approx(f1, abs=2)
    assert d2 == pytest.approx(d1, abs=0.1)


def test_cancelar_conserva_los_segmentos_ya_hechos(
        sample_video, tmp_path, ffmpeg, jobs_dir):
    cancel = threading.Event()
    opts = base_opts(sample_video, tmp_path, segment_seconds=1)

    def progress(done, _total):
        if done >= 40:          # ya paso el primer segmento (30 frames)
            cancel.set()

    with pytest.raises(ProcessingCancelled):
        VideoProcessor().run(opts, progress=progress, log=lambda *_: None,
                             cancel_event=cancel)

    pendientes = find_resumable_jobs()
    assert pendientes, "deberia quedar trabajo reanudable"
    assert any(s.done for s in pendientes[0].segments)
    assert pendientes[0].pending


def test_reanudar_completa_el_video(sample_video, tmp_path, ffmpeg, jobs_dir):
    cancel = threading.Event()
    opts = base_opts(sample_video, tmp_path, segment_seconds=1)

    def progress(done, _total):
        if done >= 40:
            cancel.set()

    with pytest.raises(ProcessingCancelled):
        VideoProcessor().run(opts, progress=progress, log=lambda *_: None,
                             cancel_event=cancel)

    # Segunda pasada: mismos ajustes, sin cancelar.
    out = VideoProcessor().run(base_opts(sample_video, tmp_path, segment_seconds=1),
                               log=lambda *_: None)
    frames, dur = contar_frames(ffmpeg, out)
    assert frames == pytest.approx(60, abs=2)
    assert dur == pytest.approx(2.0, abs=0.2)
    # Al terminar, el trabajo se limpia.
    assert not find_resumable_jobs()


def test_cambiar_un_ajuste_invalida_lo_reanudable(sample_video, tmp_path, jobs_dir):
    cancel = threading.Event()

    def progress(done, _total):
        if done >= 40:
            cancel.set()

    with pytest.raises(ProcessingCancelled):
        VideoProcessor().run(base_opts(sample_video, tmp_path, segment_seconds=1),
                             progress=progress, log=lambda *_: None, cancel_event=cancel)
    assert find_resumable_jobs()

    # Otro recorte => otra huella => trabajo distinto, nada que reaprovechar.
    registro = []
    VideoProcessor().run(
        base_opts(sample_video, tmp_path, segment_seconds=1, crop_left=10, crop_right=10,
                  output_path=str(tmp_path / "otro.mp4")),
        log=registro.append)
    assert not any("Reanudando" in linea for linea in registro)


def test_el_deflicker_no_rompe_la_exportacion(sample_video, tmp_path, ffmpeg, jobs_dir):
    out = VideoProcessor().run(
        base_opts(sample_video, tmp_path, segment_seconds=0, deflicker=True),
        log=lambda *_: None)
    frames, _ = contar_frames(ffmpeg, out)
    assert frames == pytest.approx(60, abs=2)


def test_pausar_y_reanudar_no_altera_el_resultado(sample_video, tmp_path, ffmpeg,
                                                  jobs_dir):
    """Pausar no es cancelar: el video tiene que salir identico."""
    import threading
    import time

    referencia = VideoProcessor().run(
        base_opts(sample_video, tmp_path, output_path=str(tmp_path / "ref.mp4"),
                  segment_seconds=0),
        log=lambda *_: None)
    frames_ref, dur_ref = contar_frames(ffmpeg, referencia)

    pause = PauseToken()
    pausado_en: list[int] = []

    def progress(done, _total):
        # Pausar a mitad, desde otro hilo (como hace el boton de la interfaz).
        if done == 25 and not pausado_en:
            pausado_en.append(done)
            pause.pause()
            threading.Timer(0.6, pause.resume).start()

    salida = VideoProcessor().run(
        base_opts(sample_video, tmp_path, output_path=str(tmp_path / "pausado.mp4"),
                  segment_seconds=0),
        progress=progress, log=lambda *_: None, pause=pause)

    assert pausado_en, "la prueba no llego a pausar"
    frames, dur = contar_frames(ffmpeg, salida)
    assert frames == frames_ref, "pausar perdio o duplico fotogramas"
    assert dur == pytest.approx(dur_ref, abs=0.05)
    assert pause.paused_seconds >= 0.4
    assert tiene_audio(ffmpeg, salida)


def test_pausar_detiene_de_verdad_el_avance(sample_video, tmp_path, ffmpeg, jobs_dir):
    import threading
    import time

    pause = PauseToken()
    avance: list[int] = []
    pause.pause()   # arranca pausado

    def progress(done, _total):
        avance.append(done)

    resultado: list = []

    def correr():
        resultado.append(VideoProcessor().run(
            base_opts(sample_video, tmp_path, segment_seconds=0),
            progress=progress, log=lambda *_: None, pause=pause))

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    time.sleep(2.0)
    quietos = len(avance)
    assert quietos <= 1, f"avanzo {quietos} fotogramas estando en pausa"
    pause.resume()
    hilo.join(timeout=120)
    assert resultado, "no termino tras reanudar"
    assert len(avance) > quietos


def test_la_salida_queda_etiquetada_como_bt709(sample_video, tmp_path, ffmpeg, jobs_dir):
    out = VideoProcessor().run(
        base_opts(sample_video, tmp_path, segment_seconds=0), log=lambda *_: None)
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(out)],
                          capture_output=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    texto = proc.stderr.decode("utf-8", errors="replace")
    assert "bt709" in texto
