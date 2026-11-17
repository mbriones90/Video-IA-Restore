# Avisos de terceros

El codigo de Video-IA-Restore esta bajo licencia MIT (ver [LICENSE](LICENSE)).
Este fichero cubre el material de terceros que el repositorio **redistribuye** o
descarga en tiempo de ejecucion.

---

## Pesos de modelo redistribuidos en este repositorio

En `models/` se incluyen pesos convertidos al formato ncnn:

| Fichero | Origen |
|---|---|
| `realesr_general_x4v3.ncnn.param` / `.bin` | Real-ESRGAN `realesr-general-x4v3` (v0.2.5.0) |
| `RealESRGAN_x4plus_fp32.ncnn.param` / `.bin` | Real-ESRGAN `RealESRGAN_x4plus` |

Son obras derivadas de los pesos oficiales de
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), convertidas con
`tools/make_ncnn.py`, y se distribuyen bajo la misma licencia BSD de 3
clausulas que el original:

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Los pesos `.onnx` **no** se incluyen: `sync.bat` los descarga y los verifica por
SHA-256 contra `models/manifest.json`.

---

## Dependencias que se instalan o descargan, pero no se redistribuyen

Ninguna de estas viaja dentro del repositorio. Se listan porque el programa no
funciona sin ellas y conviene saber bajo que condiciones llegan.

| Componente | Licencia | Como llega |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | `sync.bat` (uv), segun el backend |
| [ncnn](https://github.com/Tencent/ncnn) | BSD-3-Clause | `sync.bat` (uv) |
| [PySide6 / Qt](https://www.qt.io/) | LGPL-3.0 | `sync.bat` (uv) |
| [OpenCV](https://opencv.org/) (`opencv-python-headless`) | Apache-2.0 | `sync.bat` (uv) |
| [NumPy](https://numpy.org/) | BSD-3-Clause | `sync.bat` (uv) |
| [FFmpeg](https://ffmpeg.org/) (via `imageio-ffmpeg`) | LGPL-2.1+ / GPL-2+ segun el build | `sync.bat` (uv) |
| [VapourSynth](https://www.vapoursynth.com/) | LGPL-2.1 | Instalacion aparte del usuario |
| [QTGMC](https://github.com/HomeOfVapourSynthEvolution) y MVTools | GPL-2.0 | `tools/install_qtgmc.py` |
| Librerias CUDA / cuDNN de NVIDIA | [NVIDIA Software License](https://docs.nvidia.com/cuda/eula/) | `sync.bat`, solo con el backend CUDA |

**Ojo con el desentrelazado:** QTGMC y MVTools son GPL-2.0. Usarlos desde este
programa para tus propios videos no tiene ninguna pega, pero si redistribuyes un
paquete que ya los traiga dentro (por ejemplo lo que genera `empaquetar.bat`),
ese paquete queda sujeto a la GPL. Por eso `install_qtgmc.py` los descarga en el
PC de cada usuario en vez de incluirlos aqui.
