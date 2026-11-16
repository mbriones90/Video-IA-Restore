"""Planificacion de segmentos, huella de ajustes y reanudacion."""

from __future__ import annotations

from pathlib import Path

import pytest

from vhs_upscaler.ffmpeg_utils import VideoInfo
from vhs_upscaler.pipeline import (
    Job, ProcessOptions, Segment, effective_duration, job_fingerprint, plan_segments,
)


def info(duration=600.0, fps=30.0) -> VideoInfo:
    return VideoInfo(Path("x.mp4"), 720, 480, fps, int(duration * fps), duration)


def opts(**kw) -> ProcessOptions:
    base = dict(input_path="x.mp4", output_path="y.mp4")
    base.update(kw)
    return ProcessOptions(**base)


# ------------------------------------------------------------- segmentos ----

def test_un_solo_segmento_si_el_video_es_mas_corto_que_el_tramo():
    segs = plan_segments(opts(segment_seconds=300), info(duration=120), 30.0)
    assert len(segs) == 1
    assert segs[0].start == 0.0
    assert segs[0].duration == pytest.approx(120.0)


def test_segment_seconds_cero_no_trocea():
    segs = plan_segments(opts(segment_seconds=0), info(duration=3600), 30.0)
    assert len(segs) == 1


def test_los_segmentos_cubren_todo_sin_huecos_ni_solapes():
    segs = plan_segments(opts(segment_seconds=300), info(duration=1000), 30.0)
    assert len(segs) == 4  # 300 + 300 + 300 + 100
    assert sum(s.duration for s in segs) == pytest.approx(1000.0)
    for anterior, siguiente in zip(segs, segs[1:]):
        assert siguiente.start == pytest.approx(anterior.start + anterior.duration)


def test_los_segmentos_respetan_el_punto_de_inicio():
    segs = plan_segments(opts(segment_seconds=100, start_seconds=60), info(duration=360), 30.0)
    assert segs[0].start == pytest.approx(60.0)
    assert sum(s.duration for s in segs) == pytest.approx(300.0)


def test_modo_prueba_limita_la_duracion():
    o = opts(start_seconds=15, max_seconds=10)
    assert effective_duration(o, info(duration=600)) == pytest.approx(10.0)


def test_el_numero_de_frames_sigue_al_fps_de_salida():
    segs = plan_segments(opts(segment_seconds=0), info(duration=10), 59.94)
    assert segs[0].frames == 599


# ------------------------------------------------------------- huella -------

def test_la_huella_cambia_si_cambia_un_ajuste_de_imagen():
    a = job_fingerprint(opts(denoise_strength=3))
    b = job_fingerprint(opts(denoise_strength=7))
    assert a != b


def test_la_huella_no_cambia_al_renombrar_la_salida():
    a = job_fingerprint(opts(output_path="uno.mp4"))
    b = job_fingerprint(opts(output_path="otro.mp4"))
    assert a == b, "renombrar la salida no debe invalidar horas de proceso"


def test_la_huella_cubre_los_ajustes_de_color_y_estabilizacion():
    base = job_fingerprint(opts())
    assert job_fingerprint(opts(color_range="full")) != base
    assert job_fingerprint(opts(color_matrix="bt709")) != base
    assert job_fingerprint(opts(deflicker=True)) != base


# ------------------------------------------------------------- estado -------

def test_el_estado_del_trabajo_va_y_vuelve_del_disco(jobs_dir):
    job = Job(job_id="abc123", input_path="in.mp4", output_path="out.mp4",
              fingerprint="f" * 16, fps_out=59.94, width=2880, height=1920,
              segments=[Segment(0, 0.0, 300.0, 17982, done=True),
                        Segment(1, 300.0, 120.0, 7193)])
    job.save()

    recargado = Job.load(job.state_path)
    assert recargado is not None
    assert recargado.job_id == "abc123"
    assert len(recargado.segments) == 2
    assert recargado.segments[0].done is True
    assert len(recargado.pending) == 1


def test_un_estado_corrupto_no_revienta(jobs_dir):
    ruta = jobs_dir / "roto" / "state.json"
    ruta.parent.mkdir(parents=True)
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert Job.load(ruta) is None


def test_un_estado_inexistente_devuelve_none(jobs_dir):
    assert Job.load(jobs_dir / "no" / "existe.json") is None
