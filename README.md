# VHS Upscaler

Aplicacion de escritorio para Windows que restaura grabaciones de VHS (capturadas con OBS)
y las escala a **1080p con IA**, sin deformar la proporcion original.

## Requisitos

- Windows 10/11 de 64 bits.
- **UV**: no hace falta instalarlo a mano. `sync.bat` lo instala oficialmente
  en la PC (usuario) la primera vez. Si prefieres instalarlo tu:
  `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.
- UV descarga Python 3.12 y las dependencias solo.

## Uso rapido

Hay DOS scripts:

1. **`sync.bat` - prepara la PC (una vez por PC, o al cambiar de backend):**
   - Instala UV oficialmente si falta.
   - Detecta el hardware (NVIDIA -> CUDA; AMD/Intel -> DirectML; otro -> CPU)
     o fuerza uno: `sync.bat cpu`, `sync.bat cuda`, `sync.bat directml`.
   - Instala las dependencias de ese backend + QTGMC (VapourSynth).
   - Descarga los modelos de IA y los plugins de QTGMC.
   - Guarda el backend en `config\backend.txt`.
2. **`iniciar.bat` - abre la aplicacion** (3 pasos: origen, configuracion,
   exportar). Solo lanza: si falta algo te dice que ejecutes `sync.bat`.

3. Dentro de la app:
   - **Paso 1 - Origen:** eliges el video grabado; ves resolucion, fps, duracion y un frame.
   - **Paso 2 - Configuracion:** preset (modo facil), modelo de IA, dispositivo,
     filtros VHS, salida y **vista previa** (original vs escalado) de un frame real.
   - **Paso 3 - Exportar:** resumen, prueba rapida, progreso con ETA, cancelar
     y abrir carpeta.

## Modo facil (presets)

No hace falta entender todas las opciones. En el **paso 2**, arriba de todo, hay un
combo con configuraciones predefinidas que ajustan todo de una vez:

| Preset | Para que sirve |
|---|---|
| **Facil - VHS estandar** | El recomendado para empezar: desentrelaza, limpia grano y color, 1080p |
| **Cinta muy ruidosa o danada** | Limpieza fuerte para cintas con mucho grano o manchas |
| **Cinta en buen estado** | Limpieza suave, conserva el detalle original |
| **Maxima calidad** | Modelo grande x4plus: mas detalle, mucho mas lento |

Flujo recomendado:

1. Elige un preset (el primero arranque aplica "Facil" solo).
2. Pulsa **"Ver como quedara el video"**: procesa UN fotograma y muestra el antes
   y el despues. Puedes **ampliar la vista previa** a pantalla casi completa.
3. Si te gusta, ve al paso 3 y usa **"Prueba rapida"** (10 segundos reales de la
   cinta) antes de lanzar el video completo.

Si cambias cualquier opcion a mano, el combo pasa a **"Personalizado"** para que
sepas que ya no estas en un preset. Todas las opciones tienen explicacion al
dejar el mouse encima (tooltip).

### Forzar un backend

```bat
iniciar.bat cpu          Siempre disponible (respaldo)
iniciar.bat cuda         PC con NVIDIA (p. ej. RTX 4070)
iniciar.bat directml     Ryzen AI Max / Radeon / Intel (Windows)
sync.bat                 Prepara la PC (instala y descarga) sin abrir la app
```

Dentro de la app, en el paso 2, puedes **cambiar el dispositivo en cualquier momento**
(Auto / CPU / CUDA / DirectML / Vulkan). Si el dispositivo elegido no esta disponible,
el programa **degrada automaticamente a CPU** y lo avisa en el registro.

## Modelos de IA

| Modelo | Tamano | Uso recomendado |
|---|---|---|
| `realesr-general-x4v3` | ~5 MB | Recomendado para VHS: rapido y buen balance calidad/velocidad |
| `realesrgan-x4plus` | ~67 MB | Maxima calidad; recomendado solo con GPU |
| `bicubic` | - | Sin IA: prueba el pipeline completo en segundos |

Los modelos se definen en `models/manifest.json` (URL + SHA-256 verificado).
El **mismo archivo ONNX sirve para CPU, CUDA y DirectML**; lo que cambia entre PCs
es el runtime que instala UV (`onnxruntime`, `onnxruntime-gpu` u `onnxruntime-directml`,
declarados en conflicto para que nunca se mezclen). Por eso el respaldo CPU funciona
siempre con el mismo archivo descargado.

Para agregar un modelo nuevo: anade una entrada al manifiesto con su URL y checksum.

## Pipeline avanzado de restauracion VHS

El paso 2 implementa el flujo profesional de calibracion de cintas:

- **Desentrelazado**: bwdif (rapido) o yadif, con **paridad de campos** explicita
  (Auto / BFF para NTSC / TFF para PAL). Si el movimiento se ve a saltos, la
  paridad esta invertida: cambiala y repite la prueba rapida.
- **QTGMC** (opcional, requiere VapourSynth): desentrelazado compensado por
  movimiento, EZDenoise para el grano magnetico, limpieza de crominancia para el
  sangrado de color, y doble tasa temporal real 29.97i -> 59.94p. Si VapourSynth
  no esta instalado, se usa bwdif como respaldo automatico.
- **Reduccion de ruido** (hqdn3d, fuerza 1-10): limpia el grano de cinta ANTES
  de escalar, con limpieza extra de crominancia activable.
- **Recorte de overscan** (opcional, multiplos de 2): para margenes laterales
  muertos y la franja de salto de cabezales. Por defecto 0 (sin recorte).
- **Correccion de color** y **debanding** para cintas desvaidas o con bandas.
- **Grano de pelicula** (opcional, intensidad 1-20): quita el aspecto "oleoso"
  de la IA sobre piel y superficies lisas. Se aplica solo a la imagen (las
  barras negras quedan limpias) y se refleja en la vista previa. En CLI:
  `--grain --grain-strength 6`.
- **Audio mono** (opcional): unifica un audio estereo irregular (un canal mas
  bajo o danado) mezclando ambos canales en uno central. En CLI: `--mono-audio`.
- **Estabilizacion de brillo** (opcional): quita el "bombeo" del AGC del
  reproductor. En CLI: `--deflicker`. Ver la seccion dedicada mas abajo.
- **Exposicion automatica** (opcional): nivela las escenas de dia y de noche
  corrigiendo la gamma por escena. En CLI: `--auto-exposure`. Ver su seccion.

## Modo prueba (calibracion obligatoria)

El paso 3 incluye **"Prueba rapida"**: exporta solo 10 segundos a partir del
segundo 15 con los ajustes actuales. Revisa el resultado (paridad, ruido, color)
y, si esta bien, lanza el procesamiento completo. En CLI: `--test`.

## Analisis automatico de la cinta

Tres cosas que antes habia que adivinar a mano y que, si se fallan, arruinan la
restauracion en silencio. El boton **"Analizar la cinta y ajustar solo"** del
paso 2 las mide en cuatro puntos del video (nunca en uno solo: el principio de
una cinta suele ser negro o barras de color) y rellena los controles:

| Que mide | Con que | Por que importa |
|---|---|---|
| Paridad de campos (TFF/BFF/progresivo) | `idet` | Si se invierte, el movimiento sale a saltos. Si el material ya es progresivo, desentrelazar borra detalle para nada |
| Bordes negros y franja de cabezales | `cropdetect` | Escalar barras negras con IA es tiempo de GPU tirado, y la franja de ruido de abajo confunde al modelo |
| Rango (16-235 / 0-255) y matriz (BT.601 / BT.709) | cabecera del archivo | Acertar mal aplasta o lava los negros, y no hay vuelta atras |

Ante la duda **no decide**: si los votos de paridad quedan repartidos, lo deja
en "Auto" en vez de arriesgarse. El recorte que propone es siempre el mas
conservador de las cuatro muestras, para no cortar imagen real porque una
escena oscura pareciera borde negro.

En CLI: `--analyze` (analiza, aplica y procesa) o `--analyze-only` (solo informa).

## Pausar y reanudar

El boton **Pausar** del paso 3 detiene el proceso donde esta, sin perder nada:
al reanudar sigue por el mismo fotograma. Util para soltar la GPU un rato
(jugar, editar video, otra cosa pesada) sin tirar horas de trabajo.

Funciona asi: el hilo que escala se queda esperando, y como las colas del
pipeline son **acotadas**, el decodificador y el codificador se frenan solos
(llenan su cola, se bloquean escribiendo en la tuberia y ahi se quedan, vivos
y sin gastar CPU). No hay nada que guardar ni que rehacer, y los subprocesos
ni se enteran. Por eso la pausa esta en **un solo punto** del pipeline:
pausar en tres sitios distintos solo abriria carreras.

No confundir con cancelar: cancelar tira el tramo a medias (conservando los ya
terminados, ver la seccion siguiente); pausar deja todo exactamente donde
estaba.

El tiempo detenido **no cuenta** para la velocidad ni para el tiempo restante,
y se informa aparte al terminar:

```
Completado: 600 frames en 0.5 min (+0.1 min en pausa) (22.2 fps)
```

En la CLI, con una consola de verdad, se pausa y reanuda con la tecla **P**
(si la salida esta redirigida a un archivo no hay teclado que leer y la opcion
se desactiva sola, sin molestar).

Cerrar la app estando en pausa funciona: se reanuda primero para que el hilo
pueda terminar, en vez de esperar al tiempo limite y matarlo.

## Reanudar un trabajo cortado

Una cinta de 2 h tarda horas. Si el proceso se corta al 80 % (apagon,
cancelacion, un error), antes se perdia todo.

Ahora el video se procesa **por tramos** (5 minutos por defecto). Cada tramo
terminado se guarda en `output/.jobs/<id>/` y, al final, se unen con copia de
flujo y se les pega el audio del original. Al volver a lanzar el mismo trabajo
con los **mismos ajustes**, se salta lo que ya estaba hecho.

- La identidad del trabajo es una huella de los ajustes que afectan a los
  pixeles. Cambiar el recorte o el denoise invalida lo guardado (correcto:
  el resultado seria otro); **renombrar el archivo de salida no**, porque eso
  no cambia ni un pixel y no debe costar horas.
- Un tramo marcado como listo cuyo archivo haya desaparecido se rehace: lo que
  hay en disco manda sobre el fichero de estado.
- Al terminar del todo, la carpeta de trabajo se borra sola.
- La "Prueba rapida" no trocea ni guarda estado: son 10 segundos.

En CLI: `--jobs` lista lo pendiente, `--no-resume` lo ignora y rehace todo,
`--segment-seconds 0` desactiva el troceado.

## Exposicion automatica (escenas de dia y de noche)

En una cinta familiar conviven escenas al sol y escenas de interior o de noche,
y un solo ajuste de brillo no sirve para las dos. La opcion **"Nivelar escenas
de dia y de noche"** mide cada fotograma y corrige la **gamma** para acercar
los medios tonos al objetivo.

**Por que gamma y no un auto-levels.** Medido sobre una cinta real de 97 min,
una escena oscura y una clara dan esto:

| escena | p1 | p25 | mediana | p75 | p99 |
|---|---|---|---|---|---|
| oscura (min 73) | 14 | 16 | **28** | 44 | 228 |
| clara (min 8) | 14 | 23 | **56** | 157 | 230 |

El punto de negro y el de blanco son **identicos**; lo unico que cambia son los
medios tonos. No hay histograma que estirar, asi que el filtro `normalize` de
FFmpeg (el "auto brillo" que recomienda todo el mundo) no solo no arregla nada:
sobre ese mismo fotograma bajo la mediana de 28 a 17, o sea que lo **empeora**.

Como esta hecho:

- La curva actua **solo entre p1 y p99**; fuera de ese tramo el valor se deja
  tal cual. Recortarlo con un clip aplastaba contra el negro todo lo que
  cayera por debajo de p1.
- **Se suaviza el exponente, no el pixel.** Si cada fotograma eligiera su gamma,
  la imagen latiria.
- **Se detectan los cortes de plano** para no arrastrar la exposicion de una
  escena a la siguiente.
- La correccion es **asimetrica**: aclarar hasta gamma 2.2, oscurecer como
  mucho un 12 %. Bajar una escena diurna al nivel de una nocturna comprimia las
  sombras y convertia la ropa oscura del fondo en un borron negro (el p1 caia
  de 10 a 1). El objetivo funciona practicamente como **suelo**: una escena
  soleada sigue pareciendo soleada.

Resultado sobre la cinta real, con intensidad al 80 %:

| minuto | mediana | accion |
|---|---|---|
| 73 (noche) | 28 -> 48 | aclara +20 |
| 89 (oscura) | 30 -> 49 | aclara +19 |
| 8 (correcta) | 56 -> 56 | no la toca |
| 40 (clara) | 138 -> 130 | apenas -8 |
| 93 (muy clara) | 205 -> 201 | apenas -4 |

Coste: **0.53 ms por fotograma** (una LUT de 256 entradas). En CLI:
`--auto-exposure`, con `--auto-exposure-target` (16-160),
`--auto-exposure-strength` (0-100) y `--auto-exposure-smoothing`.

Se ve en la vista previa, asi que puedes calibrarla sin exportar.

**Viene activada**: los cuatro presets la encienden, con la intensidad ajustada
al tipo de cinta. Aclarar una escena oscura tambien sube el grano magnetico,
asi que a mas ruido, mano mas ligera:

| preset | intensidad |
|---|---|
| Facil - VHS estandar | 70 % |
| Cinta muy ruidosa o danada | 55 % |
| Cinta en buen estado | 50 % |
| Maxima calidad | 80 % |

Si prefieres el aspecto original, basta con quitar la casilla: el combo pasa a
"Personalizado" y se respeta.

## Estabilizacion de brillo (deflicker)

El control automatico de ganancia del reproductor sube y baja el brillo medio
entre fotogramas; tras escalar con IA se nota como un parpadeo. La opcion
**"Estabilizar el brillo"** normaliza cada fotograma hacia la media movil de
los ultimos 5, con un tope del 12 % para no "corregir" un corte de plano real.

Funciona en las dos rutas (FFmpeg y QTGMC) porque se aplica en memoria, justo
antes de escalar.

**Lo que NO hace**: corregir el temblor horizontal de cabezal (time-base
error). Eso no tiene arreglo bueno sobre el video ya digitalizado; se arregla
capturando la senal RF con [vhs-decode](https://github.com/oyvindln/vhs-decode).
Preferimos decirlo a fingir que lo resuelve.

## Equivalencias con el flujo de trabajo original

| Pieza del flujo original | Estado |
|---|---|
| Orquestador CLI con ruta/destino, paridad, denoise, modo prueba | Integrado (GUI y CLI) |
| Inyeccion de entorno: `runtime\bin` antepuesto al PATH | Integrado (todos los subprocesos lo heredan) |
| IPC sin disco (tuberia stdout -> stdin) | Integrado (frames crudos en memoria, sin temporales) |
| SIGPIPE al cerrar descriptor | Equivalente Windows: cierre del stdin del codificador al terminar |
| ffms2 indexado + paridad BFF/TFF | En ruta QTGMC (VapourSynth); en ruta FFmpeg es bwdif/yadif con parity |
| QTGMC + EZDenoise + ChromaNoise + 59.94p | Integrado como backend opcional (ver abajo) |
| Recorte overscan/head-switching multiplo de 2 | Integrado (opcional, validado) |
| Smoke test 10 s desde el segundo 15 | Integrado ("Prueba rapida" y `--test`) |
| Dither 8 -> 10-bit (fmtconv) en el grafo | En ruta QTGMC (fmtconv en la conversion a RGB); en la ruta directa no hace falta: los frames viajan sin comprimir a la IA |
| Correccion de color en 16 bits | En ruta QTGMC, dentro del grafo (antes se hacia en numpy sobre RGB de 8 bits ya cuantizado, y abria bandas) |
| Deteccion automatica de paridad / overscan / rango | Integrado (`idet`, `cropdetect`, cabecera) |
| Reanudacion por tramos | Integrado (`output/.jobs/`) |

## Rendimiento: que se midio y que se dejo fuera

Todo medido en RTX 4070, fotograma VHS 720x480 escalado x4 con
`realesr-general-x4v3` (`uv run python tools/benchmark.py`):

| Cambio | ms/frame | Nota |
|---|---|---|
| Punto de partida (tiles de 256 px, fp32, etapas en serie) | 320 | |
| Un solo pase con el fotograma completo | 94 | Trocear material SD sale caro: el solape recalcula pixeles y cada tile paga el coste fijo de numpy |
| Conversiones numpy sin copias intermedias | 93 | Se escribe plano a plano en el buffer definitivo |
| Modelo en fp16 | **73** | Diferencia visible contra fp32: 2 niveles sobre 255 |

Y en el pipeline completo (export QTGMC real, 12 s):

| | fps | Nota |
|---|---|---|
| Etapas en serie | ~10 | El total era la SUMA de decodificar + escalar + codificar |
| Etapas solapadas | **~16** | Hilos unidos por colas acotadas: el total es el MAXIMO |

**Lo que se probo y no se quedo:**

- **Lotes (batch)**: agrupar 2 o 4 fotogramas no mejora nada (50.2 ms/frame con
  lote 4 frente a 49.6 con lote 1). Un fotograma SD de 1.4 Mpx que sale a
  5.5 Mpx ya satura la GPU; no hay coste de lanzamiento que amortizar. El
  modelo se deja con el lote dinamico por si algun dia se procesa material mas
  pequeno, pero el pipeline no agrupa.
- **TensorRT**: `onnxruntime` lo lista como proveedor disponible, pero cargarlo
  necesita ~2 GB de wheels de NVIDIA que no se instalan, y construye un engine
  por cada resolucion la primera vez (minutos). Se deja accesible con
  `set VHS_TENSORRT=1`, y si no esta se cae a CUDA sin fallar.
- **fp16 con entrada/salida en fp16**: 6x mas lento (308 ms). numpy no tiene
  SIMD para float16 y convertir 16.6 M de valores por fotograma cuesta mas que
  todo lo que ahorra la GPU. Por eso la variante fp16 mantiene la E/S en
  float32 (`keep_io_types=True`) y convierte **dentro** del grafo, en la GPU.

## Backend CUDA (NVIDIA) - sin instalar el CUDA Toolkit

`onnxruntime-gpu` >= 1.21 se compila contra **CUDA 13 + cuDNN 9** y no trae esas
librerias. Si faltan, el proveedor CUDA no carga (`cublasLt64_13.dll ... Error
126`) y la inferencia cae **en silencio** a CPU (30x mas lenta).

El grupo `cuda` del `pyproject.toml` las instala como wheels de NVIDIA dentro
del propio `.venv` (`onnxruntime-gpu[cuda,cudnn]`), asi que **no hace falta el
CUDA Toolkit del sistema**; solo el driver de NVIDIA. `inference.py` llama a
`ort.preload_dlls()` antes de crear la sesion para que Windows las encuentre, y
comprueba con `get_providers()` que el proveedor quedo realmente activo: si no,
avisa en el log en vez de fingir que va por GPU.

Medicion real en RTX 4070, frame VHS 720x480 escalado x4 con
`realesr-general-x4v3` (`uv run python tools/benchmark.py`):

| Dispositivo | ms/frame | fps | 1 h de cinta (29.97 fps) |
|---|---|---|---|
| **CUDA (RTX 4070)** | 92 | 10.9 | ~2.8 h |
| Vulkan / ncnn | 151 | 6.6 | ~4.5 h |
| CPU | 2662 | 0.4 | ~80 h |

En material SD el escalador procesa el **frame completo de una pasada** en vez
de trocearlo (`tile_size = 0`): es ~2x mas rapido y no hay costuras que
recomponer. Si la GPU se queda sin memoria, degrada solo a bloques de 256 px.

## Backend Vulkan (ncnn) - recomendado en graficas AMD

DirectML (ONNX Runtime) es lento en iGPU AMD. El proyecto incluye ahora el
backend **ncnn con Vulkan** (el mismo motor que usa Upscayl), que ademas activa
**fp16** en la GPU. Mediciones reales en la Radeon 8060S (Ryzen AI Max 395),
frame VHS 720x480 escalado x4 con `realesr-general-x4v3`:

| Dispositivo | ms/frame | fps | 1 h de cinta (29.97 fps) |
|---|---|---|---|
| **Vulkan / ncnn** | 213 | 4.9 | ~6.4 h |
| DirectML | 280 | 3.6 | ~8.4 h |
| CPU | 1755 | 0.6 | ~52 h |

Como funciona:
- El paquete `ncnn` se instala en TODAS las PCs (es base, no grupo), porque
  convive sin conflicto con cualquier onnxruntime.
- Los pesos ncnn (`.ncnn.param`/`.ncnn.bin`, fp16) de ambos modelos **viajan con
  el proyecto** en `models/`; si actualizas los ONNX, regeneralos con:
  `uv run python tools/make_ncnn.py`
- El dispositivo "Automatico" ahora prefiere **CUDA > Vulkan > DirectML > CPU**.
- ROCm en Windows no es oficial para esta iGPU; si algun dia lo es, se puede
  anadir como grupo igual que los demas.

## QTGMC (VapourSynth) - INSTALADO AUTOMATICAMENTE

El desentrelazado profesional QTGMC (compensado por movimiento, doble tasa
temporal 29.97i -> 59.94p, denoise con MVTools) lo instala `sync.bat`:
sincroniza el grupo `qtgmc` de UV (VapourSynth R82 + plugins pip: ffms2,
fmtconv, znedi3, mvtools, mvutensils, resize2) y descarga los dos plugins que
no estan en pip (RemoveGrainVS, TTempSmooth). Es idempotente: la primera vez
descarga todo; despues tarda segundos.

- `vspipe.exe` queda en `.venv\Scripts` (lo encuentra la app sola).
- mvutensils va fijado a v9: vsdeinterlace 0.9.3 es incompatible con v10.
- Verificado: QTGMC procesa a ~154 fps en el Ryzen AI Max 395.
- Si la descarga de QTGMC falla (p. ej. sin internet), `sync.bat` lo avisa
  y la app arranca igual con bwdif de respaldo; reintenta despues con:
  `uv run python tools\install_qtgmc.py`
- Sin instalarlo nunca, la app sigue usando bwdif sin problema.

## Resoluciones de salida y proporcion

Alturas disponibles: **720p, 1080p, 1440p y 4K (2160p)**, cada una en dos variantes
que se calculan con la proporcion real del video cargado. Ejemplo con VHS 4:3:

| Variante | 1080p | 4K | Efecto |
|---|---|---|---|
| **Proporcion nativa** | 1440x1080 | 2880x2160 | Conserva el lienzo original: **SIN barras**, no modifica nada |
| **Lienzo 16:9** | 1920x1080 | 3840x2160 | **Modifica el lienzo** agregando barras negras laterales |

Si tu video ya es 16:9, la variante 16:9 coincide con la nativa y no agrega barras.
**Ninguna opcion estira ni recorta la imagen**: solo cambia el lienzo de salida.

## Ancho de banda del encode (Mbps)

En el paso 2, "Modo de tasa" permite elegir:
- **CRF (calidad constante):** el tamano final varia segun el contenido.
- **Bitrate objetivo en Mbps:** tamano predecible. El pico se limita a 1.5x.

Referencias orientativas: 1080p ~ 8-16 Mbps, 1440p ~ 16-24 Mbps, 4K ~ 32-50 Mbps.
El audio siempre va en AAC 192 kbps.

## Linea de comandos (opcional)

```bat
uv run python -m vhs_upscaler.cli "C:\ruta\video.mkv" --device directml
uv run python -m vhs_upscaler.cli "video.mkv" --model bicubic --device cpu
uv run python -m vhs_upscaler.cli "video.mkv" --height 2160 --canvas native --bitrate-mbps 40
uv run python -m vhs_upscaler.cli "video.mkv" --height 1080 --canvas 169 --bitrate-mbps 12
REM Prueba rapida con paridad NTSC y recorte de overscan:
uv run python -m vhs_upscaler.cli "video.mkv" --test --parity bff --crop 8,0,8,6
REM Analizar la cinta (paridad, overscan, color), aplicar y procesar:
uv run python -m vhs_upscaler.cli "video.mkv" --analyze --deint qtgmc
REM Solo analizar, sin tocar nada:
uv run python -m vhs_upscaler.cli "video.mkv" --analyze-only
REM Nivelar escenas de dia y de noche:
uv run python -m vhs_upscaler.cli "video.mkv" --auto-exposure
REM Mas suave, y apuntando a una imagen algo mas clara:
uv run python -m vhs_upscaler.cli "video.mkv" --auto-exposure --auto-exposure-strength 60 --auto-exposure-target 65
REM Estabilizar el brillo y trocear en tramos de 10 min:
uv run python -m vhs_upscaler.cli "video.mkv" --deflicker --segment-seconds 600
REM Durante el proceso, pulsa P para pausar o reanudar.
REM Ver que trabajos quedaron a medias:
uv run python -m vhs_upscaler.cli "video.mkv" --jobs
```

## Pruebas

```bat
uv run --group dev pytest                 REM todo
uv run --group dev pytest -m "not slow"   REM solo lo que corre en cualquier PC
```

Las marcadas `slow` generan un clip sintetico con FFmpeg y exportan de verdad
(con el modelo bicubico, sin IA): comprueban que el troceado no duplica ni
pierde fotogramas, que cancelar conserva lo ya hecho y que reanudar completa el
video. Las demas no necesitan nada instalado.

El fallo de la imagen repetida 9 veces tiene su propia prueba
(`tests/test_frames.py`): el contrato es que un buffer planar GBR de vspipe
salga como BGR entrelazado con los canales en su sitio.

## Estructura

```
sync.bat                 Prepara la PC: UV, backend, dependencias, modelos, QTGMC
iniciar.bat              Lanzador: abre la app con el backend guardado por sync.bat
pyproject.toml           Dependencias base y grupos cpu/cuda/directml (UV)
models/manifest.json     Catalogo de modelos con URL y SHA-256
config/config.local.json Preferencias de ESTA PC (se genera solo)
src/vhs_upscaler/
  app.py                 GUI de 3 pasos (PySide6)
  pipeline.py            Decodificar -> IA -> codificar, solapado y por tramos
  inference.py           Motores de escalado (ONNX / ncnn / bicubico)
  model_opt.py           Variante fp16 del modelo, generada y cacheada sola
  analysis.py            Paridad (idet), overscan (cropdetect) y color
  vs_backend.py          Grafo VapourSynth + QTGMC (color en 16 bits)
  logging_setup.py       Log rotativo a output/logs/
  hardware.py            Deteccion y respaldo automatico a CPU
  models.py              Descarga selectiva y verificacion de modelos
  ffmpeg_utils.py        FFmpeg incluido (imageio-ffmpeg) + lectura de SAR
  cli.py                 Uso sin interfaz grafica
tests/                   pytest (ver seccion "Pruebas")
output/                  Videos exportados
output/logs/             Registro rotativo (5 ficheros de 2 MB)
output/.jobs/            Tramos de trabajos a medias (se borran al terminar)
```

## Notas y limites actuales

- Se procesa **un video a la vez**, a proposito.
- El escalado hace **un solo pase con el fotograma completo**; solo si la GPU se
  queda sin memoria degrada a tiles de 256 px con solape.
- Las tres etapas (decodificar, escalar, codificar) corren **a la vez** en hilos
  distintos unidos por colas cortas.
- El audio se conserva del archivo original (AAC 192k); si no hay audio, no pasa nada.
- FFmpeg viene incluido via `imageio-ffmpeg`; si tienes `ffmpeg` en el PATH o en
  `runtime\bin` se prefiere ese.
- Con QTGMC activo no aplican la interpolacion de fps ni el debanding (QTGMC ya
  duplica la tasa temporal); la correccion de color se hace **en 16 bits dentro
  del grafo**, no en memoria.
- El fps que se le pasa al codificador en la ruta QTGMC sale de `vspipe --info`,
  no del redondeo de OpenCV: con 29.97 mal redondeado a 30.00, una cinta de 2 h
  se desincroniza varios segundos.
- Si el archivo trae el pixel marcado como no cuadrado (NTSC 720x480 con
  SAR 8:9), se corrige antes de encajar en el lienzo. Sin eso la imagen saldria
  estirada un 12 %.
- La salida se etiqueta siempre como **BT.709, rango limitado**, tanto en el
  flujo como en la cabecera del contenedor: nada queda a que lo adivine el
  reproductor.
- Todo lo que sale en el cuadro de registro queda tambien en
  `output/logs/vhs_upscaler.log` (5 ficheros de 2 MB, rotativos).
- Requiere Windows 10/11 de 64 bits. UV descarga Python 3.12 automaticamente.
