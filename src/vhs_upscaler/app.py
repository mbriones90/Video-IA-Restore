"""Interfaz de escritorio: pipeline de 3 pasos (Origen -> Configuracion -> Exportar)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from .config import AppConfig
from .ffmpeg_utils import probe_video, read_frame
from .hardware import DEVICE_LABELS, detect_hardware, resolve_device
from .models import BICUBIC_ID, BICUBIC_NAME, is_downloaded, load_manifest
from .paths import CONFIG_PATH, OUTPUT_DIR, ensure_dirs
from .presets import CUSTOM_KEY, PRESETS
from .resolutions import build_output_sizes
from .logging_setup import LOG_PATH, log_to, setup_logging
from .pipeline import (
    EXPORT_FORMATS, ProcessOptions, ProcessingCancelled, VideoProcessor,
    quality_value,
    PauseToken, add_grain, build_auto_exposure, extract_filtered_frame,
    find_resumable_jobs, fit_to_canvas,
)
from .vs_backend import qtgmc_available

DARK_STYLE = """
QMainWindow, QWidget { background: #1e1f24; color: #e6e6e6; font-size: 10pt; }
QGroupBox { border: 1px solid #3a3d46; border-radius: 6px; margin-top: 12px;
            padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #2f6fdb; border: none; border-radius: 5px;
              padding: 8px 16px; color: white; }
QPushButton:hover { background: #3d7ef0; }
QPushButton:disabled { background: #44464f; color: #9a9ca5; }
QPushButton#secondary { background: #3a3d46; }
QPushButton#secondary:hover { background: #4a4d58; }
QPushButton#danger { background: #b03a3a; }
QLineEdit, QComboBox, QPlainTextEdit { background: #2a2c33; border: 1px solid #3a3d46;
    border-radius: 4px; padding: 5px; selection-background-color: #2f6fdb; }
QProgressBar { background: #2a2c33; border: 1px solid #3a3d46; border-radius: 5px;
               text-align: center; height: 20px; }
QProgressBar::chunk { background: #2f6fdb; border-radius: 4px; }
QLabel#title { font-size: 15pt; font-weight: bold; }
QLabel#muted { color: #9a9ca5; }
QSlider::groove:horizontal { height: 4px; background: #3a3d46; }
QSlider::handle:horizontal { background: #2f6fdb; width: 14px; margin: -6px 0;
                             border-radius: 7px; }
"""


def ndarray_to_pixmap(frame: np.ndarray, max_size=(460, 260)) -> QPixmap:
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(
        max_size[0], max_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class ProcessWorker(QThread):
    progressed = Signal(int, int)
    logged = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, opts: ProcessOptions):
        super().__init__()
        self.opts = opts
        self.cancel_event = threading.Event()
        self.pause = PauseToken()

    def run(self):
        try:
            out = VideoProcessor().run(
                self.opts,
                progress=lambda d, t: self.progressed.emit(d, t),
                # log_to duplica cada linea al fichero rotativo: si un proceso
                # de 3 h falla al final, el cuadro de texto ya no lo tiene todo.
                log=log_to(self.logged.emit, "proceso"),
                cancel_event=self.cancel_event,
                pause=self.pause,
            )
            self.succeeded.emit(str(out))
        except ProcessingCancelled:
            self.failed.emit("CANCELADO")
        except Exception as exc:
            self.failed.emit(str(exc))


class AnalysisWorker(QThread):
    """Mide paridad de campos, overscan y color sin bloquear la interfaz."""

    ready = Signal(object)   # analysis.Analysis
    logged = Signal(str)
    failed = Signal(str)

    def __init__(self, path: str, info):
        super().__init__()
        self.path = path
        self.info = info
        self.cancel_event = threading.Event()

    def run(self):
        try:
            from .analysis import analyze

            result = analyze(self.path, self.info, progress=self.logged.emit)
            if not self.cancel_event.is_set():
                self.ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class PreviewWorker(QThread):
    ready = Signal(object, object)  # original filtrado, escalado (ambos en lienzo 1080p)
    logged = Signal(str)
    failed = Signal(str)

    def __init__(self, opts: ProcessOptions, second: float):
        super().__init__()
        self.opts = opts
        self.second = second
        self.cancel_event = threading.Event()

    def run(self):
        try:
            from .inference import create_upscaler

            self.logged.emit("Generando vista previa (1 frame)...")
            frame = extract_filtered_frame(self.opts, self.second)
            if self.opts.auto_exposure:
                # Un solo fotograma: no hay historial que suavizar, pero el
                # gamma que se calcula es el mismo que usaria el export.
                frame = build_auto_exposure(self.opts).apply(frame)
            if self.cancel_event.is_set():
                return
            upscaler = create_upscaler(self.opts.model_id, self.opts.device,
                                       log=self.logged.emit, fp16=self.opts.fp16)
            if self.cancel_event.is_set():
                return
            sr = upscaler.upscale(frame)
            tw, th = self.opts.target_width, self.opts.target_height
            orig = fit_to_canvas(frame, tw, th)
            fin = fit_to_canvas(sr, tw, th)
            if self.opts.grain:
                orig = add_grain(orig, self.opts.grain_strength)
                fin = add_grain(fin, self.opts.grain_strength)
            self.ready.emit(orig, fin)
        except Exception as exc:
            self.failed.emit(str(exc))


class SourcePage(QWidget):
    """Paso 1: elegir el archivo grabado con OBS y ver su informacion."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        title = QLabel("Paso 1 de 3 - Origen del video")
        title.setObjectName("title")
        layout.addWidget(title)
        hint = QLabel("Selecciona el archivo que grabaste con OBS desde tu capturadora VHS.\n"
                      "Se procesa un video a la vez.")
        hint.setObjectName("muted")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Ningun archivo seleccionado...")
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Seleccionar video...")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        self.info_label = QLabel("")
        layout.addWidget(self.info_label)
        self.preview = QLabel("La vista previa aparecera aqui")
        self.preview.setObjectName("muted")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setFrameShape(QFrame.StyledPanel)
        layout.addWidget(self.preview, 1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Elegir video de VHS", "",
            "Videos (*.mp4 *.mkv *.avi *.mov *.ts *.m2ts *.mpg *.mpeg);;Todos (*.*)"
        )
        if not path:
            return
        try:
            info = probe_video(path)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Video no valido", str(exc))
            return
        self.window.input_path = path
        self.window.video_info = info
        self.window.config_page.refresh_sizes(info.width, info.height, info.sar)
        self.path_edit.setText(path)
        self.info_label.setText(info.summary())
        try:
            frame = read_frame(path, second=info.duration / 2)
            self.preview.setPixmap(ndarray_to_pixmap(frame, (640, 360)))
        except RuntimeError:
            pass
        self.window.refresh_navigation()


class ConfigPage(QWidget):
    """Paso 2: modelo, dispositivo, filtros VHS y salida."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        outer = QVBoxLayout(self)
        title = QLabel("Paso 2 de 3 - Configuracion")
        title.setObjectName("title")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        outer.addWidget(scroll, 1)
        cfg = window.config
        self._applying = False

        # --- Modo facil: presets explicados ---
        preset_box = QGroupBox("Paso 1: elige una configuracion predefinida (modo facil)")
        preset_layout = QVBoxLayout(preset_box)
        hint = QLabel(
            "Consejo: elige un preset, pulsa 'Generar vista previa' (al final de esta "
            "pagina) para ver como quedara, y si te convence exporta. Solo toca las "
            "opciones de abajo si quieres afinar algo."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        preset_layout.addWidget(hint)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Personalizado (tus cambios manuales)", CUSTOM_KEY)
        for key, preset in PRESETS.items():
            self.preset_combo.addItem(preset["label"], key)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.preset_combo.setToolTip(
            "Los presets ajustan TODAS las opciones de una vez segun el estado de tu cinta. "
            "Si luego cambias cualquier opcion a mano, vuelve a 'Personalizado'."
        )
        preset_layout.addWidget(self.preset_combo)
        self.preset_desc = QLabel("Elige un preset para ver su explicacion.")
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setObjectName("muted")
        preset_layout.addWidget(self.preset_desc)
        layout.addWidget(preset_box)

        # --- Modelo y dispositivo ---
        ai_box = QGroupBox("Escalado con IA")
        ai_form = QFormLayout(ai_box)
        self.model_combo = QComboBox()
        for m in load_manifest():
            estado = "" if is_downloaded(m) else "  [no descargado]"
            self.model_combo.addItem(m.name + estado, m.id)
        self.model_combo.addItem(BICUBIC_NAME, BICUBIC_ID)
        idx = self.model_combo.findData(cfg.model_id)
        self.model_combo.setCurrentIndex(max(0, idx))
        self.model_combo.currentIndexChanged.connect(self._update_model_desc)
        self.model_combo.setToolTip(
            "Modelo de IA que inventa el detalle faltante al escalar.\n"
            "- General x4v3: rapido, ideal para VHS.\n"
            "- x4plus: mas detalle pero MUCHO mas lento.\n"
            "- Bicubico: sin IA, solo para probar que todo funciona."
        )
        ai_form.addRow("Modelo:", self.model_combo)
        self.model_desc = QLabel("")
        self.model_desc.setWordWrap(True)
        self.model_desc.setObjectName("muted")
        ai_form.addRow("", self.model_desc)

        self.device_combo = QComboBox()
        for dev in ("auto", "cpu", "cuda", "directml", "vulkan"):
            label = DEVICE_LABELS[dev]
            if dev in ("cuda", "directml", "vulkan") and not window.hardware.device_available(dev):
                label += "  (no disponible en esta PC)"
            self.device_combo.addItem(label, dev)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(cfg.device)))
        self.device_combo.setToolTip(
            "Donde se ejecuta la IA. 'Automatico' elige lo mas rapido de esta PC:\n"
            "GPU NVIDIA (CUDA) > Vulkan/ncnn (rapido en AMD) > DirectML > CPU.\n"
            "En graficas AMD se recomienda Vulkan: suele ser varias veces mas rapido\n"
            "que DirectML. La CPU siempre funciona como respaldo."
        )
        ai_form.addRow("Dispositivo:", self.device_combo)
        gpu_note = "GPU NVIDIA detectada: " + window.hardware.nvidia_gpu if window.hardware.nvidia_gpu \
            else "Sin GPU NVIDIA: se usara DirectML (si esta instalado) o CPU."
        note = QLabel(gpu_note + "\nLa CPU siempre queda como respaldo automatico.")
        note.setObjectName("muted")
        ai_form.addRow("", note)
        layout.addWidget(ai_box)

        # --- Filtros VHS ---
        vhs_box = QGroupBox("Limpieza de cinta VHS (antes de escalar)")
        vhs = QGridLayout(vhs_box)

        self.analyze_btn = QPushButton("Analizar la cinta y ajustar solo")
        self.analyze_btn.setObjectName("secondary")
        self.analyze_btn.clicked.connect(self._analyze)
        self.analyze_btn.setToolTip(
            "Mide la cinta en varios puntos y rellena por ti:\n"
            "- Paridad de campos (si se invierte, el movimiento sale a saltos).\n"
            "- Bordes negros y franja de cabezales (recorte de overscan).\n"
            "- Rango y matriz de color (negros aplastados o lavados).\n"
            "Tarda unos segundos y no modifica el video."
        )
        vhs.addWidget(self.analyze_btn, 0, 0, 1, 2)
        self.analyze_result = QLabel("")
        self.analyze_result.setWordWrap(True)
        self.analyze_result.setObjectName("muted")
        vhs.addWidget(self.analyze_result, 1, 0, 1, 2)

        self.deint_check = QCheckBox("Desentrelazar (quita rayas de campos entrelazados)")
        self.deint_check.setChecked(cfg.deinterlace)
        self.deint_check.setToolTip(
            "El VHS guarda la imagen en dos medios cuadros. Sin esto se ven rayas "
            "horizontales en el movimiento. Dejalo activado salvo que el video ya "
            "venga desentrelazado."
        )
        self.deint_combo = QComboBox()
        self.deint_combo.addItem("bwdif (rapido, buena calidad)", "bwdif")
        self.deint_combo.addItem("yadif (mas rapido)", "yadif")
        qtgmc_label = "QTGMC (calidad profesional, requiere VapourSynth)"
        if not qtgmc_available():
            qtgmc_label += "  [no instalado: usara bwdif]"
        self.deint_combo.addItem(qtgmc_label, "qtgmc")
        self.deint_combo.setToolTip(
            "Como se reconstruyen los cuadros completos:\n"
            "- bwdif: buena calidad y rapido (recomendado).\n"
            "- yadif: el mas rapido.\n"
            "- QTGMC: calidad profesional con suavizado de movimiento; requiere "
            "instalar VapourSynth (si no esta, se usa bwdif)."
        )
        idx = {"bwdif": 0, "yadif": 1}.get(cfg.deinterlace_method, 2)
        self.deint_combo.setCurrentIndex(idx)
        vhs.addWidget(self.deint_check, 2, 0)
        vhs.addWidget(self.deint_combo, 2, 1)

        self.parity_combo = QComboBox()
        self.parity_combo.addItem("Auto (detectar)", "auto")
        self.parity_combo.addItem("BFF - campo inferior primero (NTSC)", "bff")
        self.parity_combo.addItem("TFF - campo superior primero (PAL)", "tff")
        self.parity_combo.setCurrentIndex(max(0, self.parity_combo.findData(cfg.field_parity)))
        self.parity_combo.setToolTip("Si el movimiento se ve a saltos tras exportar, la paridad esta invertida: cambiala aqui.")
        vhs.addWidget(QLabel("Paridad de campos:"), 3, 0)
        vhs.addWidget(self.parity_combo, 3, 1)

        self.double_fps_check = QCheckBox("Doble tasa temporal 29.97i -> 59.94p (solo QTGMC)")
        self.double_fps_check.setChecked(cfg.qtgmc_double_fps)
        self.double_fps_check.setToolTip(
            "Duplica los fotogramas de salida con movimiento compensado. "
            "Mas fluido, pero el escalado trabaja al doble de frames."
        )
        vhs.addWidget(self.double_fps_check, 4, 0, 1, 2)

        crop_widget = QWidget()
        crop_row = QHBoxLayout(crop_widget)
        crop_row.setContentsMargins(0, 0, 0, 0)
        self.crop_spins = {}
        for name, label in (("crop_left", "Izq"), ("crop_top", "Arr"),
                            ("crop_right", "Der"), ("crop_bottom", "Aba")):
            spin = QSpinBox()
            spin.setRange(0, 200)
            spin.setSingleStep(2)
            spin.setValue(getattr(cfg, name))
            spin.setToolTip("Solo overscan / franja de cabezales. Multiplos de 2. 0 = sin recorte.")
            self.crop_spins[name] = spin
            crop_row.addWidget(QLabel(label))
            crop_row.addWidget(spin)
        vhs.addWidget(QLabel("Recorte overscan (px):"), 5, 0)
        vhs.addWidget(crop_widget, 5, 1)

        self.denoise_check = QCheckBox("Reduccion de ruido de cinta (hqdn3d / EZDenoise)")
        self.denoise_check.setChecked(cfg.denoise)
        self.denoise_check.setToolTip(
            "Elimina el grano/fondo de la cinta ANTES de escalar. Si desactivas esto, "
            "la IA amplificara el ruido y quedara peor."
        )
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(1, 10)
        self.denoise_slider.setValue(cfg.denoise_strength)
        vhs.addWidget(self.denoise_check, 6, 0)
        vhs.addWidget(self._labeled_slider(self.denoise_slider, "suave", "fuerte"), 6, 1)

        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItem("hqdn3d (rapido, suficiente)", "hqdn3d")
        self.denoise_method_combo.addItem(
            "MVTools: degrain compensado por movimiento (mejor en cinta)", "mvtools")
        self.denoise_method_combo.setCurrentIndex(
            max(0, self.denoise_method_combo.findData(cfg.denoise_method)))
        self.denoise_method_combo.setToolTip(
            "hqdn3d promedia pixeles en el mismo sitio: rapido, pero arrastra "
            "estelas detras de lo que se mueve.\n\n"
            "MVTools sigue cada bloque de un fotograma al siguiente y promedia "
            "a lo largo de su trayectoria: el grano de cinta, que no tiene "
            "trayectoria, se cancela, y el detalle real se queda. Es la misma "
            "limpieza que usa QTGMC por dentro.\n\n"
            "Si tu capturadora ya desentrelaza por hardware y no puedes "
            "desactivarlo, esta es la forma de aprovechar lo bueno de QTGMC "
            "sin volver a desentrelazar, que estropearia la imagen."
        )
        vhs.addWidget(QLabel("Metodo de limpieza:"), 16, 0)
        vhs.addWidget(self.denoise_method_combo, 16, 1)
        self.denoise_method_note = QLabel("")
        self.denoise_method_note.setWordWrap(True)
        self.denoise_method_note.setObjectName("muted")
        vhs.addWidget(self.denoise_method_note, 17, 0, 1, 2)
        self.deint_check.stateChanged.connect(self._refresh_denoise_method)
        self.deint_combo.currentIndexChanged.connect(self._refresh_denoise_method)
        self.denoise_method_combo.currentIndexChanged.connect(
            self._refresh_denoise_method)

        self.chroma_check = QCheckBox("Limpieza de crominancia (sangrado de color VHS)")
        self.chroma_check.setChecked(cfg.chroma_denoise)
        self.chroma_check.setToolTip(
            "El VHS suele 'manchar' los colores hacia los lados (bordes de colores "
            "fantasma). Esto limpia ese sangrado sin perder nitidez."
        )
        vhs.addWidget(self.chroma_check, 7, 0, 1, 2)

        self.color_check = QCheckBox("Correccion de color y brillo")
        self.color_check.setChecked(cfg.color_correct)
        self.color_check.setToolTip(
            "Para cintas desvanidas o oscuras: ajusta brillo, contraste y saturacion "
            "con los controles de abajo. Activalo solo si la vista previa lo necesita."
        )
        vhs.addWidget(self.color_check, 8, 0, 1, 2)
        self.brightness = self._make_slider(-100, 100, cfg.brightness)
        self.contrast = self._make_slider(0, 200, cfg.contrast)
        self.saturation = self._make_slider(0, 200, cfg.saturation)
        vhs.addWidget(QLabel("Brillo"), 9, 0)
        vhs.addWidget(self.brightness, 9, 1)
        vhs.addWidget(QLabel("Contraste"), 10, 0)
        vhs.addWidget(self.contrast, 10, 1)
        vhs.addWidget(QLabel("Saturacion"), 11, 0)
        vhs.addWidget(self.saturation, 11, 1)

        self.deband_check = QCheckBox("Debanding (suaviza bandas en degradados)")
        self.deband_check.setChecked(cfg.deband)
        self.deband_check.setToolTip(
            "Quita las 'escaleras' de color visibles en cielos o paredes lisas. "
            "Activalo si la vista previa muestra bandas."
        )
        vhs.addWidget(self.deband_check, 12, 0, 1, 2)

        self.deflicker_check = QCheckBox("Estabilizar el brillo (quita el 'bombeo' del AGC)")
        self.deflicker_check.setChecked(cfg.deflicker)
        self.deflicker_check.setToolTip(
            "El reproductor sube y baja el brillo solo entre fotogramas; tras "
            "escalar se nota como un parpadeo. Esto lo suaviza.\n"
            "No corrige el temblor horizontal de cabezal: eso solo se arregla "
            "capturando la senal RF (vhs-decode)."
        )
        vhs.addWidget(self.deflicker_check, 13, 0, 1, 2)

        self.autoexp_check = QCheckBox(
            "Nivelar escenas de dia y de noche (exposicion automatica)")
        self.autoexp_check.setChecked(cfg.auto_exposure)
        self.autoexp_check.setToolTip(
            "En una cinta familiar hay escenas al sol y escenas de interior o de "
            "noche, y la misma correccion no sirve para las dos.\n\n"
            "Esto mide cada fotograma y corrige la GAMMA para acercar los medios "
            "tonos al objetivo, dejando el negro y el blanco donde estaban (no "
            "quema las altas luces ni lava los negros).\n\n"
            "Detecta los cortes de plano para no arrastrar la exposicion de una "
            "escena a la siguiente, y suaviza la correccion entre fotogramas "
            "para que la imagen no lata."
        )
        vhs.addWidget(self.autoexp_check, 18, 0, 1, 2)

        self.autoexp_target = self._make_slider(16, 160, cfg.auto_exposure_target)
        self.autoexp_target.setToolTip(
            "Mediana de luma a la que se lleva cada escena. 56 es el valor medido "
            "en escenas bien expuestas de una cinta familiar tipica. Subelo si "
            "todo queda oscuro; bajalo si las noches pierden ambiente."
        )
        vhs.addWidget(QLabel("Brillo objetivo:"), 19, 0)
        vhs.addWidget(self._labeled_slider(self.autoexp_target, "oscuro", "claro"), 19, 1)

        self.autoexp_strength = self._make_slider(0, 100, cfg.auto_exposure_strength)
        self.autoexp_strength.setToolTip(
            "Cuanto de la correccion calculada se aplica. Al 100 % todas las "
            "escenas quedan al mismo nivel; por debajo se conserva parte de la "
            "diferencia original entre dia y noche, que suele verse mas natural."
        )
        vhs.addWidget(QLabel("Intensidad:"), 20, 0)
        vhs.addWidget(self._labeled_slider(self.autoexp_strength, "sutil", "total"), 20, 1)

        self.range_combo = QComboBox()
        self.range_combo.addItem("Automatico (leer del archivo)", "auto")
        self.range_combo.addItem("Limitado 16-235 (TV, lo habitual)", "limited")
        self.range_combo.addItem("Completo 0-255 (PC)", "full")
        self.range_combo.setCurrentIndex(max(0, self.range_combo.findData(cfg.color_range)))
        self.range_combo.setToolTip(
            "Como interpretar los niveles del ORIGEN. Si se acierta mal, los "
            "negros salen aplastados o lavados y ya no hay vuelta atras. "
            "Dejalo en automatico salvo que el analisis diga otra cosa."
        )
        vhs.addWidget(QLabel("Rango de color del origen:"), 21, 0)
        vhs.addWidget(self.range_combo, 21, 1)

        self.matrix_combo = QComboBox()
        self.matrix_combo.addItem("Automatico (segun la resolucion)", "auto")
        self.matrix_combo.addItem("BT.601 / SMPTE 170M (SD, VHS)", "bt601")
        self.matrix_combo.addItem("BT.709 (HD)", "bt709")
        self.matrix_combo.setCurrentIndex(max(0, self.matrix_combo.findData(cfg.color_matrix)))
        self.matrix_combo.setToolTip(
            "Matriz de conversion del ORIGEN. Una captura de VHS es SD, es "
            "decir BT.601; usar BT.709 por error tine los colores."
        )
        vhs.addWidget(QLabel("Matriz de color del origen:"), 22, 0)
        vhs.addWidget(self.matrix_combo, 22, 1)
        layout.addWidget(vhs_box)

        # --- Salida ---
        out_box = QGroupBox("Salida")
        out_form = QFormLayout(out_box)
        self.size_combo = QComboBox()
        self.size_combo.setToolTip(
            "Tamano del video final.\n"
            "- 'Proporcion nativa': conserva el formato original de la cinta (4:3), "
            "sin barras negras.\n"
            "- 'Lienzo 16:9': formato de TV moderna; agrega barras negras a los lados "
            "para llenar la pantalla SIN estirar la imagen."
        )
        out_form.addRow("Resolucion:", self.size_combo)
        ratio_note = QLabel(
            "Ninguna opcion estira la imagen. 'Proporcion nativa' conserva el lienzo original "
            "(SIN barras). 'Lienzo 16:9' MODIFICA el lienzo agregando barras negras para "
            "completar el formato panoramico."
        )
        ratio_note.setWordWrap(True)
        ratio_note.setObjectName("muted")
        out_form.addRow("", ratio_note)
        self.fps_combo = QComboBox()
        self.fps_combo.addItem("FPS originales (recomendado)", 0)
        self.fps_combo.addItem("Interpolar a 50 fps (movimiento mas fluido, lento)", 50)
        self.fps_combo.addItem("Interpolar a 60 fps (movimiento mas fluido, lento)", 60)
        self.fps_combo.setCurrentIndex(max(0, self.fps_combo.findData(cfg.interpolate_fps)))
        self.fps_combo.setToolTip(
            "Los fotogramas por segundo del resultado. Deja 'originales' salvo que "
            "quieras movimiento mas fluido (inventa fotogramas intermedios: es lento "
            "y puede verse artificial en cintas)."
        )
        out_form.addRow("Fotogramas:", self.fps_combo)
        self.format_combo = QComboBox()
        for key, fmt in EXPORT_FORMATS.items():
            self.format_combo.addItem(fmt["label"], key)
        legacy = {"h264": "mp4_h264", "h265": "mp4_h265"}.get(cfg.codec, "")
        selected = cfg.export_format if cfg.export_format in EXPORT_FORMATS else legacy
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(selected or "mp4_h264")))
        self.format_combo.setToolTip(
            "Formato del archivo final.\n"
            "- MP4 H.264: funciona en cualquier telefono, TV o PC. El mas seguro\n"
            "  para mandarle el video a un familiar.\n"
            "- MP4/MKV H.265: misma calidad con archivo mas pequeno.\n"
            "- MP4/MKV AV1 por GPU: entre un 13% y un 39% menos de peso\n"
            "  que H.264 a igual calidad, y sale casi gratis porque lo\n"
            "  codifica un chip aparte de la tarjeta, no los nucleos que\n"
            "  hacen el escalado. Pide GeForce RTX 40 o superior; si no la\n"
            "  hay, baja a H.265 avisando. Es la mejor opcion para guardar\n"
            "  el maestro de la cinta.\n"
            "- MP4 AV1 por CPU: comprime algo mejor, pero tan lento que\n"
            "  frena todo el proceso. Solo para un PC sin RTX 40.\n"
            "- WebM VP9: ideal para subir a webs.\n"
            "AV1 lo reproduce cualquier TV, telefono o navegador de 2020 en\n"
            "adelante, pero no un reproductor viejo."
        )
        out_form.addRow("Formato:", self.format_combo)
        self.format_note = QLabel("")
        self.format_note.setWordWrap(True)
        self.format_note.setObjectName("muted")
        out_form.addRow("", self.format_note)
        self.quality_combo = QComboBox()
        for q, desc in (("alta", "Alta"), ("media", "Media"),
                        ("rapida", "Rapida (archivo menor)")):
            self.quality_combo.addItem(desc, q)
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(cfg.quality)))
        self.quality_combo.setToolTip(
            "Calidad del archivo final (solo aplica en modo CRF). Alta = mejor "
            "imagen y archivo mas grande."
        )
        out_form.addRow("Calidad:", self.quality_combo)
        self.grain_check = QCheckBox("Grano de pelicula ligero (anti 'oleoso')")
        self.grain_check.setChecked(cfg.grain)
        self.grain_check.setToolTip(
            "Agrega un grano fino tipo pelicula sobre el video final para que la "
            "imagen no se vea 'oleosa' o de plastico tras el escalado con IA. "
            "Deja las barras negras limpias. Se refleja en la vista previa."
        )
        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(1, 20)
        self.grain_slider.setValue(cfg.grain_strength)
        self.grain_slider.setToolTip("4-8 es sutil (recomendado). 12 o mas se nota bastante.")
        out_form.addRow("Acabado:", self.grain_check)
        out_form.addRow("Intensidad de grano:", self._labeled_slider(self.grain_slider, "sutil", "fuerte"))
        self.mono_check = QCheckBox("Unificar audio a MONO (un solo canal central)")
        self.mono_check.setChecked(cfg.audio_mono)
        self.mono_check.setToolTip(
            "Mezcla los dos canales (izquierdo + derecho) en un solo canal central. "
            "Util si tu captura VHS suena 'irregular' o de lado a lado porque un "
            "canal esta mas bajo o danado. El volumen se mantiene equilibrado."
        )
        out_form.addRow("Audio:", self.mono_check)
        self.format_combo.currentIndexChanged.connect(self._toggle_rate_controls)
        self._toggle_rate_controls()

        dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(cfg.output_dir)
        dir_btn = QPushButton("Examinar...")
        dir_btn.setObjectName("secondary")
        dir_btn.clicked.connect(self._pick_output_dir)
        dir_row.addWidget(self.output_dir_edit, 1)
        dir_row.addWidget(dir_btn)
        out_form.addRow("Carpeta destino:", dir_row)
        layout.addWidget(out_box)

        # --- Proceso ---
        proc_box = QGroupBox("Proceso (velocidad y reanudacion)")
        proc_form = QFormLayout(proc_box)
        self.segment_spin = QSpinBox()
        self.segment_spin.setRange(0, 60)
        self.segment_spin.setSuffix(" min")
        self.segment_spin.setSpecialValueText("Sin trocear (no se puede reanudar)")
        self.segment_spin.setValue(cfg.segment_minutes)
        self.segment_spin.setToolTip(
            "El video se procesa por tramos y cada uno se guarda al terminar. "
            "Si el proceso se corta (apagon, cancelacion, error), al volver a "
            "lanzarlo se reanuda desde el ultimo tramo listo en vez de empezar "
            "de cero. 5 minutos va bien para una cinta entera."
        )
        proc_form.addRow("Tamano de tramo:", self.segment_spin)
        self.resume_check = QCheckBox("Reanudar trabajos a medias en vez de empezar de cero")
        self.resume_check.setChecked(cfg.resume)
        proc_form.addRow("", self.resume_check)
        self.fp16_check = QCheckBox("Media precision (fp16) en GPU: ~1.3x mas rapido")
        self.fp16_check.setChecked(cfg.fp16)
        self.fp16_check.setToolTip(
            "Usa una variante del modelo que calcula en 16 bits. En una RTX 4070 "
            "baja de 93 a 73 ms por fotograma y la diferencia en la imagen es de "
            "2 niveles sobre 255 (invisible). Se desactiva solo en CPU."
        )
        proc_form.addRow("", self.fp16_check)
        layout.addWidget(proc_box)

        # --- Vista previa ---
        prev_box = QGroupBox("Vista previa de un frame (antes de exportar)")
        prev_layout = QVBoxLayout(prev_box)
        self.preview_btn = QPushButton("Ver como quedara el video (vista previa)")
        self.preview_btn.clicked.connect(self._make_preview)
        self.preview_btn.setToolTip(
            "Procesa UN solo fotograma con los ajustes actuales y muestra el antes "
            "y el despues. Es gratis y rapido: usalo antes de procesar todo el video."
        )
        prev_layout.addWidget(self.preview_btn)
        enlarge_btn = QPushButton("Ampliar vista previa")
        enlarge_btn.setObjectName("secondary")
        enlarge_btn.clicked.connect(self._enlarge_preview)
        prev_layout.addWidget(enlarge_btn)
        pair = QHBoxLayout()
        self.prev_original = QLabel("Original (con filtros)")
        self.prev_scaled = QLabel("Escalado")
        for lbl in (self.prev_original, self.prev_scaled):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(460, 260)
            lbl.setFrameShape(QFrame.StyledPanel)
        pair.addWidget(self.prev_original)
        pair.addWidget(self.prev_scaled)
        prev_layout.addLayout(pair)
        layout.addWidget(prev_box)
        self._update_model_desc()
        self._refresh_denoise_method()
        self._refresh_format_note()
        self.format_combo.currentIndexChanged.connect(self._refresh_format_note)
        self.quality_combo.currentIndexChanged.connect(self._refresh_format_note)
        self.size_combo.currentIndexChanged.connect(self._refresh_format_note)
        self.refresh_sizes()

        # Cualquier cambio manual devuelve el combo de presets a "Personalizado".
        for widget in (
            self.model_combo, self.device_combo, self.size_combo, self.fps_combo,
            self.format_combo, self.quality_combo,
            self.deint_combo, self.parity_combo, self.range_combo, self.matrix_combo,
            self.denoise_method_combo,
        ):
            widget.currentIndexChanged.connect(self._mark_custom)
        for widget in (
            self.deint_check, self.double_fps_check, self.denoise_check,
            self.chroma_check, self.color_check, self.deband_check,
            self.grain_check, self.mono_check, self.deflicker_check,
            self.autoexp_check,
        ):
            widget.stateChanged.connect(self._mark_custom)
        for widget in (
            self.denoise_slider, self.brightness, self.contrast, self.saturation,
            self.grain_slider,
            self.autoexp_target, self.autoexp_strength,
        ):
            widget.valueChanged.connect(self._mark_custom)
        for spin in self.crop_spins.values():
            spin.valueChanged.connect(self._mark_custom)

    def refresh_sizes(self, src_w: int = 720, src_h: int = 480, sar: float = 1.0):
        """Recalcula las resoluciones segun la proporcion VISIBLE del video."""
        current = self.size_combo.currentData()
        selected = current.key if current else self.window.config.output_size
        self.size_combo.clear()
        for s in build_output_sizes(src_w, src_h, sar):
            self.size_combo.addItem(s.label, s)
        for i in range(self.size_combo.count()):
            if self.size_combo.itemData(i).key == selected:
                self.size_combo.setCurrentIndex(i)
                return
        self.size_combo.setCurrentIndex(0)

    def _toggle_rate_controls(self):
        fmt = EXPORT_FORMATS.get(self.format_combo.currentData(), EXPORT_FORMATS["mp4_h264"])
        self.quality_combo.setEnabled(fmt["rate"])

    @staticmethod
    def _make_slider(lo: int, hi: int, value: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        return s

    @staticmethod
    def _labeled_slider(slider: QSlider, left: str, right: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(left))
        lay.addWidget(slider, 1)
        lay.addWidget(QLabel(right))
        return w

    def _apply_preset(self):
        key = self.preset_combo.currentData()
        if not key or key == CUSTOM_KEY:
            return
        preset = PRESETS[key]
        self.preset_desc.setText(preset["desc"])
        values = preset["values"]
        self._applying = True
        try:
            self._set_combo(self.model_combo, values.get("model_id"))
            self.deint_check.setChecked(values.get("deinterlace", True))
            self._set_combo(self.deint_combo, values.get("deinterlace_method", "bwdif"))
            self._set_combo(self.parity_combo, values.get("field_parity", "auto"))
            self.double_fps_check.setChecked(values.get("qtgmc_double_fps", True))
            self.denoise_check.setChecked(values.get("denoise", True))
            self.denoise_slider.setValue(values.get("denoise_strength", 3))
            self._set_combo(self.denoise_method_combo,
                            values.get("denoise_method", "hqdn3d"))
            self.chroma_check.setChecked(values.get("chroma_denoise", True))
            self.color_check.setChecked(values.get("color_correct", False))
            self.deband_check.setChecked(values.get("deband", False))
            self.grain_check.setChecked(values.get("grain", False))
            self.grain_slider.setValue(values.get("grain_strength", 6))
            # Los presets fijan TODOS los controles, tambien los que no les
            # interesan: si un preset no nombra una opcion, se deja en su valor
            # por defecto en vez de arrastrar lo del preset anterior.
            self.deflicker_check.setChecked(values.get("deflicker", False))
            self.autoexp_check.setChecked(values.get("auto_exposure", False))
            self.autoexp_target.setValue(values.get("auto_exposure_target", 56))
            self.autoexp_strength.setValue(values.get("auto_exposure_strength", 70))
            self.mono_check.setChecked(values.get("audio_mono", False))
            self._set_combo(self.fps_combo, values.get("interpolate_fps", 0))

            self._set_combo(self.quality_combo, values.get("quality", "alta"))
            self._set_combo(self.format_combo, values.get("export_format", "mp4_h264"))

            for i in range(self.size_combo.count()):
                if self.size_combo.itemData(i).key == values.get("output_size", "1080_169"):
                    self.size_combo.setCurrentIndex(i)
                    break
            self._update_model_desc()
            self._toggle_rate_controls()
        finally:
            self._applying = False

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _mark_custom(self, *args) -> None:
        if not self._applying and self.preset_combo.currentData() != CUSTOM_KEY:
            self.preset_combo.setCurrentIndex(0)

    def _enlarge_preview(self) -> None:
        if getattr(self, "_preview_full", None) is None:
            QMessageBox.information(self, "Sin vista previa",
                                    "Primero genera la vista previa con el boton de arriba.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Vista previa ampliada (izquierda: original, derecha: escalado)")
        dlg.resize(1840, 580)
        lay = QHBoxLayout(dlg)
        for title, img in zip(("Original (con filtros VHS)", "Escalado por IA"),
                              self._preview_full):
            col = QVBoxLayout()
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignCenter)
            col.addWidget(caption)
            lbl = QLabel()
            lbl.setPixmap(ndarray_to_pixmap(img, (880, 500)))
            col.addWidget(lbl)
            lay.addLayout(col)
        dlg.exec()

    def _update_model_desc(self):
        model_id = self.model_combo.currentData()
        if model_id == BICUBIC_ID:
            self.model_desc.setText("Sin IA: util para probar el pipeline completo en segundos.")
            return
        for m in load_manifest():
            if m.id == model_id:
                estado = "Descargado." if is_downloaded(m) else \
                    "NO descargado: ejecuta iniciar.bat para bajarlo."
                self.model_desc.setText(f"{m.description}\n{estado}")
                return

    def _pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta de salida", self.output_dir_edit.text())
        if d:
            self.output_dir_edit.setText(d)

    def _refresh_format_note(self, *_):
        """Explica el cuantizador real y el peso que cabe esperar.

        Dos cosas que confunden si no se dicen: que "Alta" es CRF 17 en H.264
        y cq 26 en AV1 (el mismo nivel, escalas distintas), y que el numero
        sube solo al elegir mas resolucion.
        """
        from .pipeline import CODEC_CRF, EXPORT_FORMATS, quality_value

        clave = self.format_combo.currentData() or "mp4_h264"
        fmt = EXPORT_FORMATS.get(clave, EXPORT_FORMATS["mp4_h264"])
        codec = fmt["video"]
        calidad = self.quality_combo.currentData() or "alta"
        size = self.size_combo.currentData()
        alto = size.height if size else 1080
        if not fmt["rate"]:
            self.format_note.setText("")
            return

        q = quality_value(codec, calidad, alto)
        partes = []
        if codec == "av1_nvenc":
            partes.append(
                f"AV1 por GPU, cq {q}: el mismo nivel que CRF "
                f"{quality_value('libx264', calidad, alto)} en H.264, con un "
                f"archivo bastante menor. Su escala va de 0 a 63, asi que el "
                f"numero no se compara con el de H.264.")
        elif codec == "libaom-av1":
            partes.append(
                f"AV1 por CPU, CRF {q}. Comprime un ~12% mejor que el de GPU, "
                f"pero baja el ritmo del proceso completo de unos 15 a unos "
                f"11 fps: es el unico codec que estorba.")
        else:
            partes.append(f"{codec}, CRF {q}.")

        base = CODEC_CRF.get(codec, CODEC_CRF["libx264"]).get(calidad, q)
        if q != base:
            partes.append(
                f"Subido {int(q) - int(base)} puntos desde {base} porque a "
                f"{alto}p la imagen aguanta mas compresion sin que se note.")

        esperado = self._expected_mbps(codec, calidad, alto)
        if esperado:
            partes.append(f"En material como el tuyo a 60 fps, unos "
                          f"{esperado:g} Mbps.")
        self.format_note.setText(" ".join(partes))

    # Bitrate medido exportando 10 s reales con el pipeline completo (captura
    # VHS de 640x480 a 60 fps, escalada con realesr-general-x4v3), con el
    # cuantizador de cada perfil ya ajustado a la resolucion. Orientativo: el
    # contenido manda, y a 30 fps sale bastante por debajo.
    _MBPS_AV1_GPU = {
        720: {"alta": 9, "media": 6, "rapida": 4.5},
        1080: {"alta": 16, "media": 11, "rapida": 8},
        1440: {"alta": 21, "media": 14, "rapida": 10},
        2160: {"alta": 31, "media": 21, "rapida": 15},
    }

    def _expected_mbps(self, codec: str, calidad: str, alto: int):
        if codec != "av1_nvenc":
            return None
        cercano = min(self._MBPS_AV1_GPU, key=lambda h: abs(h - alto))
        if abs(cercano - alto) > 200:
            return None
        return self._MBPS_AV1_GPU[cercano].get(calidad)

    def _refresh_denoise_method(self, *_):
        """Avisa junto al control cuando MVTools no cabe con lo elegido."""
        from .pipeline import ProcessOptions
        from .vs_backend import effective_denoise_method

        sonda = ProcessOptions(
            input_path="", output_path="", denoise=True,
            denoise_method=self.denoise_method_combo.currentData() or "hqdn3d",
            deinterlace=self.deint_check.isChecked(),
            deinterlace_method=self.deint_combo.currentData() or "bwdif",
        )
        aviso = effective_denoise_method(sonda)[1]
        self.denoise_method_note.setText(aviso)
        self.denoise_method_note.setStyleSheet("color: #e8a33d;" if aviso else "")

    # ---------- Analisis automatico ----------
    def _analyze(self):
        if not self.window.input_path or not self.window.video_info:
            QMessageBox.information(self, "Falta el video",
                                    "Primero selecciona un video en el paso 1.")
            return
        if self.window.analysis_worker is not None:
            return
        self.analyze_btn.setEnabled(False)
        self.analyze_result.setText("Analizando la cinta...")
        worker = AnalysisWorker(self.window.input_path, self.window.video_info)
        worker.finished.connect(self.window.release_analysis_worker)
        worker.finished.connect(worker.deleteLater)
        self.window.analysis_worker = worker
        worker.logged.connect(self.window.append_log)
        worker.ready.connect(self._apply_analysis)
        worker.failed.connect(self._analysis_failed)
        worker.start()

    def _apply_analysis(self, result):
        """Vuelca lo medido en los controles, sin pisar decisiones del usuario."""
        info = self.window.video_info
        self._applying = True   # no marcar "Personalizado" por esto
        try:
            if result.parity.parity == "progressive":
                self.deint_check.setChecked(False)
            elif result.parity.interlaced:
                self.deint_check.setChecked(True)
                if self.parity_combo.currentData() == "auto":
                    self._set_combo(self.parity_combo, result.parity.parity)
            if result.crop.any:
                for name, value in (("crop_left", result.crop.left),
                                    ("crop_top", result.crop.top),
                                    ("crop_right", result.crop.right),
                                    ("crop_bottom", result.crop.bottom)):
                    self.crop_spins[name].setValue(value)
            if self.range_combo.currentData() == "auto":
                self._set_combo(self.range_combo, result.color.range)
            if self.matrix_combo.currentData() == "auto":
                self._set_combo(self.matrix_combo, result.color.matrix)
        finally:
            self._applying = False
        texto = result.summary(info)
        self.analyze_result.setText(texto + "\n(Ya aplicado a los controles de arriba.)")
        self.window.append_log("Analisis de la cinta:\n" + texto)
        self.analyze_btn.setEnabled(True)

    def _analysis_failed(self, msg: str):
        self.analyze_btn.setEnabled(True)
        self.analyze_result.setText("No se pudo analizar: " + msg)

    def enable_analyze_button(self):
        self.analyze_btn.setEnabled(True)

    def _make_preview(self):
        if not self.window.input_path:
            QMessageBox.information(self, "Falta el video", "Primero selecciona un video en el paso 1.")
            return
        opts = self.window.collect_options()
        second = (self.window.video_info.duration / 2) if self.window.video_info else 0.0
        if self.window.preview_worker is not None:
            return  # ya hay una vista previa en curso (el boton se reactiva al terminar)
        self.preview_btn.setEnabled(False)
        worker = PreviewWorker(opts, second)
        # Al terminar: soltar la referencia ANTES de deleteLater. Si no, queda un
        # wrapper de Python apuntando a un QThread ya destruido en C++ y el
        # siguiente uso revienta con "Internal C++ object already deleted".
        worker.finished.connect(self.window.release_preview_worker)
        worker.finished.connect(worker.deleteLater)
        self.window.preview_worker = worker
        worker.logged.connect(self.window.append_log)
        worker.ready.connect(self._show_preview)
        worker.failed.connect(self._preview_failed)
        worker.start()

    def enable_preview_button(self):
        self.preview_btn.setEnabled(True)

    def _show_preview(self, original, scaled):
        self._preview_full = (original, scaled)
        self.prev_original.setPixmap(ndarray_to_pixmap(original))
        self.prev_scaled.setPixmap(ndarray_to_pixmap(scaled))
        self.preview_btn.setEnabled(True)

    def _preview_failed(self, msg: str):
        self.preview_btn.setEnabled(True)
        QMessageBox.warning(self, "Error en vista previa", msg)


class ExportPage(QWidget):
    """Paso 3: resumen, progreso y exportacion del video."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        title = QLabel("Paso 3 de 3 - Exportar")
        title.setObjectName("title")
        layout.addWidget(title)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.start_btn = QPushButton("Iniciar procesamiento completo")
        self.start_btn.clicked.connect(lambda: self.window.start_processing(test=False))
        layout.addWidget(self.start_btn)

        self.test_btn = QPushButton("Prueba rapida (10 s desde el segundo 15)")
        self.test_btn.setObjectName("secondary")
        self.test_btn.clicked.connect(lambda: self.window.start_processing(test=True))
        self.test_btn.setToolTip(
            "Exporta solo un segmento corto con los ajustes actuales para validar "
            "paridad, filtros y calidad antes de procesar la cinta completa."
        )
        layout.addWidget(self.test_btn)

        self.resume_note = QLabel("")
        self.resume_note.setWordWrap(True)
        self.resume_note.setObjectName("muted")
        self.resume_note.setVisible(False)
        layout.addWidget(self.resume_note)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("Listo para comenzar.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        row = QHBoxLayout()
        self.pause_btn = QPushButton("Pausar")
        self.pause_btn.setObjectName("secondary")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.window.toggle_pause)
        self.pause_btn.setToolTip(
            "Detiene el proceso donde esta, sin perder nada. El decodificador "
            "y el codificador se quedan esperando: al reanudar sigue por el "
            "mismo fotograma.\n\n"
            "Util para liberar la GPU un rato (jugar, editar video) sin tirar "
            "horas de trabajo. El tiempo en pausa no cuenta para la velocidad "
            "ni para el tiempo restante."
        )
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.window.cancel_processing)
        self.open_btn = QPushButton("Abrir carpeta de salida")
        self.open_btn.setObjectName("secondary")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output)
        row.addWidget(self.pause_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.open_btn)
        layout.addLayout(row)

    def refresh_resume_note(self):
        """Avisa de trabajos a medias que se reanudarian al pulsar exportar."""
        try:
            jobs = find_resumable_jobs()
        except OSError:
            jobs = []
        if not jobs:
            self.resume_note.setVisible(False)
            return
        lineas = []
        for job in jobs:
            hechos = sum(1 for seg in job.segments if seg.done)
            lineas.append(f"  - {Path(job.input_path).name}: "
                          f"{hechos} de {len(job.segments)} tramos ya procesados")
        self.resume_note.setText(
            "Hay trabajo a medias guardado. Si exportas con los MISMOS ajustes, "
            "se reanuda desde donde se quedo:\n" + "\n".join(lineas)
        )
        self.resume_note.setVisible(True)

    def _open_output(self):
        import os
        os.startfile(self.window.last_output_dir)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Video-IA-Restore - Restaurador de cintas VHS con IA")
        self.resize(1100, 820)
        self.config = AppConfig.load()
        self.hardware = detect_hardware()
        self.input_path: str | None = None
        self.video_info = None
        self.worker: ProcessWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.last_output_dir = self.config.output_dir
        self._started_at = 0.0
        # Frames que ya estaban hechos al reanudar: no cuentan para la velocidad.
        self._resume_offset = 0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.pages = QStackedWidget()
        self.source_page = SourcePage(self)
        self.config_page = ConfigPage(self)
        self.export_page = ExportPage(self)
        for p in (self.source_page, self.config_page, self.export_page):
            self.pages.addWidget(p)
        layout.addWidget(self.pages, 1)

        # Primer arranque (sin config guardada): aplicar el preset facil.
        if not CONFIG_PATH.exists():
            idx = self.config_page.preset_combo.findData("facil")
            if idx >= 0:
                self.config_page.preset_combo.setCurrentIndex(idx)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("Atras")
        self.back_btn.setObjectName("secondary")
        self.back_btn.clicked.connect(lambda: self.pages.setCurrentIndex(self.pages.currentIndex() - 1))
        self.next_btn = QPushButton("Siguiente")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)
        self.refresh_navigation()

    # ---------- Navegacion ----------
    def refresh_navigation(self):
        idx = self.pages.currentIndex()
        ocupado = self.worker is not None
        self.back_btn.setEnabled(idx > 0 and not ocupado)
        self.next_btn.setVisible(idx < 2)
        self.next_btn.setEnabled(idx != 0 or self.input_path is not None)
        self.next_btn.setText("Siguiente" if idx == 0 else "Revisar y exportar")

    def _next(self):
        idx = self.pages.currentIndex()
        if idx == 0 and not self.input_path:
            return
        if idx == 1:
            self.config.save()
            self.export_page.summary.setText(self._summary_text())
            self.export_page.refresh_resume_note()
        self.pages.setCurrentIndex(idx + 1)
        self.refresh_navigation()

    def _summary_text(self) -> str:
        opts = self.collect_options()
        device, aviso = resolve_device(opts.device, self.hardware)
        size = self.config_page.size_combo.currentData()
        proporcion = ("proporcion nativa, SIN barras" if size and size.native
                      else "lienzo 16:9, barras negras si aplica (sin estirar)")
        tasa = f"calidad {opts.quality} (constante)"
        metodo = opts.deinterlace_method.upper()
        if opts.deinterlace and metodo in ("BWDIF", "YADIF"):
            metodo += f" (paridad {opts.field_parity})"
        if opts.deinterlace and metodo == "QTGMC":
            metodo += " (59.94p)" if opts.qtgmc_double_fps else " (29.97p)"
        crop = ", ".join(f"{k}={v}" for k, v in
                         (("izq", opts.crop_left), ("arr", opts.crop_top),
                          ("der", opts.crop_right), ("aba", opts.crop_bottom)) if v)
        lines = [
            f"Video: {Path(opts.input_path).name}",
            f"Salida: {opts.output_path}",
            f"Modelo: {opts.model_id}  |  Dispositivo: {DEVICE_LABELS.get(device, device)}",
            f"Resolucion: {opts.target_width}x{opts.target_height} - {proporcion}",
            f"Formato: {EXPORT_FORMATS.get(opts.export_format, {}).get('label', opts.export_format)}"
            f"  |  Tasa: {tasa}  |  Audio: {'mono' if opts.audio_mono else 'original'}",
            f"Desentrelazado: {metodo if opts.deinterlace else 'desactivado'}"
            + (f"  |  Recorte: {crop}" if crop else ""),
            f"Filtros: denoise={opts.denoise} (chroma={opts.chroma_denoise}), "
            f"color={opts.color_correct}, deband={opts.deband}, "
            f"grano={opts.grain} ({opts.grain_strength if opts.grain else '-'}), "
            f"exposicion_auto={opts.auto_exposure}"
            + (f" (objetivo {opts.auto_exposure_target}, "
               f"{opts.auto_exposure_strength}%)" if opts.auto_exposure else "") + ", "
            f"fps_interp={opts.interpolate_fps or 'no'}",
        ]
        if aviso:
            lines.append("Aviso: " + aviso)
        return "\n".join(lines)

    def collect_options(self) -> ProcessOptions:
        cp = self.config_page
        cfg = self.config
        cfg.device = cp.device_combo.currentData()
        cfg.model_id = cp.model_combo.currentData()
        cfg.output_dir = cp.output_dir_edit.text() or str(OUTPUT_DIR)
        cfg.export_format = cp.format_combo.currentData()
        cfg.quality = cp.quality_combo.currentData()
        size = cp.size_combo.currentData()
        cfg.output_size = size.key
        cfg.deinterlace = cp.deint_check.isChecked()
        cfg.deinterlace_method = cp.deint_combo.currentData()
        cfg.field_parity = cp.parity_combo.currentData()
        cfg.qtgmc_double_fps = cp.double_fps_check.isChecked()
        for name, spin in cp.crop_spins.items():
            setattr(cfg, name, spin.value())
        cfg.chroma_denoise = cp.chroma_check.isChecked()
        cfg.denoise = cp.denoise_check.isChecked()
        cfg.denoise_strength = cp.denoise_slider.value()
        cfg.denoise_method = cp.denoise_method_combo.currentData()
        cfg.color_correct = cp.color_check.isChecked()
        cfg.brightness = cp.brightness.value()
        cfg.contrast = cp.contrast.value()
        cfg.saturation = cp.saturation.value()
        cfg.deband = cp.deband_check.isChecked()
        cfg.grain = cp.grain_check.isChecked()
        cfg.grain_strength = cp.grain_slider.value()
        cfg.audio_mono = cp.mono_check.isChecked()
        cfg.interpolate_fps = cp.fps_combo.currentData()
        cfg.color_range = cp.range_combo.currentData()
        cfg.color_matrix = cp.matrix_combo.currentData()
        cfg.deflicker = cp.deflicker_check.isChecked()
        cfg.auto_exposure = cp.autoexp_check.isChecked()
        cfg.auto_exposure_target = cp.autoexp_target.value()
        cfg.auto_exposure_strength = cp.autoexp_strength.value()
        cfg.segment_minutes = cp.segment_spin.value()
        cfg.resume = cp.resume_check.isChecked()
        cfg.fp16 = cp.fp16_check.isChecked()
        stem = Path(self.input_path).stem if self.input_path else "video"
        ext = EXPORT_FORMATS.get(cfg.export_format, EXPORT_FORMATS["mp4_h264"])["ext"]
        return ProcessOptions(
            input_path=self.input_path or "",
            output_path=str(Path(cfg.output_dir) / f"{stem}_{size.width}x{size.height}.{ext}"),
            model_id=cfg.model_id, device=cfg.device,
            export_format=cfg.export_format, quality=cfg.quality,
            deinterlace=cfg.deinterlace, deinterlace_method=cfg.deinterlace_method,
            field_parity=cfg.field_parity,
            crop_left=cfg.crop_left, crop_right=cfg.crop_right,
            crop_top=cfg.crop_top, crop_bottom=cfg.crop_bottom,
            chroma_denoise=cfg.chroma_denoise, qtgmc_double_fps=cfg.qtgmc_double_fps,
            denoise=cfg.denoise, denoise_strength=cfg.denoise_strength,
            denoise_method=cfg.denoise_method,
            color_correct=cfg.color_correct, brightness=cfg.brightness,
            contrast=cfg.contrast, saturation=cfg.saturation,
            deband=cfg.deband, grain=cfg.grain, grain_strength=cfg.grain_strength,
            audio_mono=cfg.audio_mono,
            interpolate_fps=cfg.interpolate_fps,
            target_width=size.width, target_height=size.height,
            source_sar=self.video_info.sar if self.video_info else 1.0,
            color_range=cfg.color_range, color_matrix=cfg.color_matrix,
            deflicker=cfg.deflicker, deflicker_window=cfg.deflicker_window,
            auto_exposure=cfg.auto_exposure,
            auto_exposure_target=cfg.auto_exposure_target,
            auto_exposure_strength=cfg.auto_exposure_strength,
            auto_exposure_smoothing=cfg.auto_exposure_smoothing,
            fp16=cfg.fp16, resume=cfg.resume,
            segment_seconds=cfg.segment_minutes * 60,
        )

    # ---------- Procesamiento ----------
    def start_processing(self, test: bool = False):
        opts = self.collect_options()
        if test:
            opts.start_seconds = float(self.config.test_start)
            opts.max_seconds = float(self.config.test_secs)
            # La prueba son 10 s: trocear y guardar estado no aporta nada y
            # ensuciaria la lista de trabajos reanudables.
            opts.segment_seconds = 0
            opts.resume = False
            out = Path(opts.output_path)
            opts.output_path = str(out.with_name(out.stem + "_prueba" + out.suffix))
        device, aviso = resolve_device(opts.device, self.hardware)
        opts.device = device
        if aviso:
            self.append_log(aviso)
        self.config.save()
        self.last_output_dir = str(Path(opts.output_path).parent)
        self.export_page.log.clear()
        self._set_busy(True)
        self.worker = ProcessWorker(opts)
        self._started_at = time.monotonic()
        self._resume_offset = 0
        self.worker.progressed.connect(self._on_progress)
        self.worker.logged.connect(self.append_log)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._release_worker)
        self.worker.start()

    def _release_worker(self):
        """Libera el QThread SOLO cuando el hilo ya termino (senal finished).

        Soltarlo antes (p. ej. al recibir 'succeeded', emitida desde dentro de
        run()) destruye el QThread mientras el hilo aun vive y Qt aborta el
        proceso: "QThread: Destroyed while thread is still running" (0xc0000409).
        """
        w, self.worker = self.worker, None
        if w is not None:
            w.deleteLater()
        self.refresh_navigation()

    def _set_busy(self, busy: bool):
        ep = self.export_page
        ep.start_btn.setEnabled(not busy)
        ep.test_btn.setEnabled(not busy)
        ep.cancel_btn.setEnabled(busy)
        ep.pause_btn.setEnabled(busy)
        if not busy:
            ep.pause_btn.setText("Pausar")
        ep.open_btn.setEnabled(False)
        self.back_btn.setEnabled(not busy)

    def toggle_pause(self):
        if self.worker is None:
            return
        pausado = self.worker.pause.toggle()
        self.export_page.pause_btn.setText("Reanudar" if pausado else "Pausar")
        if pausado:
            self.export_page.status.setText(
                "EN PAUSA - el trabajo esta guardado donde estaba. "
                "Pulsa 'Reanudar' para seguir.")
            self.append_log("Procesamiento en pausa.")
        else:
            self.append_log(
                f"Reanudado tras {self._hms(self.worker.pause.paused_seconds)} en pausa.")

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.export_page.status.setText(
                "Cancelando... (los tramos ya terminados se conservan para reanudar)")

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, resto = divmod(seconds, 3600)
        m, s = divmod(resto, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _on_progress(self, done: int, total: int):
        pct = int(done * 100 / max(total, 1))
        self.export_page.progress.setValue(pct)
        if self.worker is not None and self.worker.pause.is_paused:
            return  # en pausa no se refresca el texto: lo puso toggle_pause
        # El tiempo detenido no cuenta: si no, tras una pausa de media hora la
        # velocidad y el tiempo restante saldrian absurdos.
        detenido = self.worker.pause.paused_seconds if self.worker else 0.0
        elapsed = time.monotonic() - self._started_at - detenido
        partes = [f"{done}/{total} frames ({pct}%)"]
        if done > 0 and elapsed > 1.0:
            # La velocidad se mide sobre los frames de ESTA ejecucion: al
            # reanudar, los tramos ya hechos no cuentan como tiempo invertido.
            hechos = max(done - self._resume_offset, 1)
            fps = hechos / elapsed
            partes.append(f"{fps:.1f} fps")
            partes.append(f"transcurrido {self._hms(elapsed)}")
            if fps > 0:
                partes.append(f"faltan {self._hms(max(total - done, 0) / fps)}")
        if detenido > 1.0:
            partes.append(f"{self._hms(detenido)} en pausa")
        self.export_page.status.setText("  |  ".join(partes))

    def release_analysis_worker(self):
        """Suelta el worker de analisis cuando termina (igual que la vista previa)."""
        self.analysis_worker = None
        self.config_page.enable_analyze_button()

    def _on_success(self, output: str):
        self._reset_worker()
        self.export_page.refresh_resume_note()
        self.export_page.progress.setValue(100)
        self.export_page.status.setText(f"Completado: {output}")
        self.export_page.open_btn.setEnabled(True)
        QMessageBox.information(self, "Listo", f"Video exportado correctamente:\n{output}")

    def _on_failed(self, msg: str):
        self._reset_worker()
        self.export_page.refresh_resume_note()
        if msg == "CANCELADO":
            self.export_page.status.setText("Procesamiento cancelado por el usuario.")
        else:
            self.export_page.status.setText("Error en el procesamiento.")
            QMessageBox.critical(self, "Error", msg)
        self.append_log(msg)

    def release_preview_worker(self):
        """Suelta la referencia al worker de vista previa cuando termina.

        Se conecta a 'finished' antes que deleteLater, asi nunca queda un
        wrapper de Python sobre un objeto C++ destruido.
        """
        self.preview_worker = None
        self.config_page.enable_preview_button()

    def _reset_worker(self):
        # self.worker se libera en _release_worker, al recibirse 'finished'.
        self._set_busy(False)
        self.refresh_navigation()

    def append_log(self, text: str):
        self.export_page.log.appendPlainText(text)

    def closeEvent(self, event):
        activos = [w for w in (self.worker,) if w is not None]
        if activos:
            resp = QMessageBox.question(
                self, "Procesamiento en curso",
                "Hay una tarea en curso. Quieres cancelar y salir?"
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return
        # Vista previa incluida: cualquier QThread vivo al destruir la app
        # provoca un abort() de Qt. Se cancela y se espera a que terminen.
        for w in (*activos, self.preview_worker, self.analysis_worker):
            if w is not None:
                w.cancel_event.set()
        # Un worker en pausa esta bloqueado esperando: hay que soltarlo o el
        # wait() de abajo agotaria los 15 s y acabaria en terminate().
        if self.worker is not None:
            self.worker.pause.resume()
        for w in (*activos, self.preview_worker, self.analysis_worker):
            if w is not None and not w.wait(15000):
                w.terminate()  # ultimo recurso: el proceso se cierra de inmediato
                w.wait(2000)
        event.accept()


def main():
    import faulthandler
    import sys

    from .paths import ROOT_DIR

    # Si algun dia la app se cierra de golpe (fallo nativo en Qt/onnxruntime/
    # ncnn), queda un volcado de pila en output/crash.log para diagnosticarlo.
    try:
        ensure_dirs()
        faulthandler.enable((ROOT_DIR / "output" / "crash.log").open("a", encoding="utf-8"))
    except OSError:
        faulthandler.enable()
    setup_logging()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.append_log(f"Registro detallado de esta sesion: {LOG_PATH}")
    window.show()
    sys.exit(app.exec())
