CARPETA RUNTIME - BINARIOS EXTERNOS OPCIONALES
================================================

Todo lo que pongas en runtime\bin se antepone automaticamente al PATH de los
subprocesos (FFmpeg, VapourSynth), asi se resuelven dependencias .dll sin
tocar el sistema.

QUE PONER AQUI
--------------

1) QTGMC (desentrelazado profesional, OPCIONAL):
   Ya NO hace falta nada manual. Ejecuta en la raiz del proyecto:

       uv run python tools\install_qtgmc.py

   Eso instala VapourSynth + todos los plugins (ffms2, fmtconv, znedi3,
   mvtools, mvutensils, resize2, RemoveGrainVS, TTempSmooth). El vspipe.exe
   queda en .venv\Scripts y la app lo detecta sola.

2) FFmpeg propio (OPCIONAL):
   Si prefieres tu propio build de FFmpeg en vez del incluido,
   pon ffmpeg.exe en runtime\bin y se usara con prioridad.

3) cache\:
   Descargas temporales del instalador de QTGMC (7zr.exe, zips). Se puede
   borrar sin problema.
