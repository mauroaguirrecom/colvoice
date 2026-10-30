# Voice Changer opcional

La generación de texto y la clonación de voz funcionan con el código principal de ColVoice. La conversión directa de una grabación usa Seed-VC como componente opcional.

Para habilitarla:

1. Descarga Seed-VC desde su repositorio oficial.
2. Revisa su licencia y sus instrucciones de instalación.
3. Coloca el código en `vendor/seed-vc/`.
4. Instala las dependencias en `.venv-vc/`.
5. Descarga sus checkpoints en `vendor/seed-vc/checkpoints/`.

ColVoice no incluye esos archivos de terceros en el repositorio público. Esto reduce el tamaño del proyecto y mantiene separadas las licencias de cada componente.
