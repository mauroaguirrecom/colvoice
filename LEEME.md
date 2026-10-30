# MAGI Eleven Studio

Abre **Iniciar-MAGI-Eleven.cmd** con doble clic. El navegador se abre cuando el modelo termina de cargar.

Dirección: http://127.0.0.1:8001

- **Audio a voz clonada:** carga una narración en **Audio original**, añade una referencia de la voz de destino y pulsa **Reemplazar voz**. Whisper transcribe el contenido en la CPU y MAGI Eleven lo vuelve a generar con la voz clonada. Puedes corregir la transcripción y conservar la duración aproximada del original.
- **Voice Changer:** convierte directamente una grabación a otra voz sin mostrar una transcripción ni volver a sintetizar el texto. Conserva la duración, el ritmo, las pausas, la emoción y la pronunciación del audio original. La referencia ideal dura 3–10 segundos; si es más larga, se recorta automáticamente para Seed-VC. La primera conversión puede tardar más mientras Seed-VC prepara sus modelos.
- **Voces guardadas:** en **Voice Clone** o **Audio a voz clonada**, carga una referencia, escribe un nombre y pulsa **Guardar voz**. Después podrás seleccionarla en **Voz guardada** sin volver a cargar el audio. Los perfiles se conservan localmente en `voices/`.
- **Voice Design:** abre la pestaña y elige uno de los perfiles precargados de Colombia para podcast o noticias. El perfil configura español, género, edad, tono, velocidad y calidad. MAGI Eleven no ofrece un control nativo de acento colombiano en Voice Design; para conservar un acento colombiano específico, usa **Voice Clone** con una referencia limpia de 3 a 10 segundos.
- **Voice Cloning:** añade un audio de referencia idealmente de 3–10 segundos. También puedes cargar una grabación larga: MAGI Eleven recorta un fragmento limpio antes de preparar el prompt. **Reference Text** es opcional: si lo dejas vacío, Whisper transcribe el mismo fragmento automáticamente; si escribes el texto, MAGI Eleven usa ese texto directamente (para audios de más de 20 segundos conviene dejarlo vacío).
- Los audios generados se pueden reproducir y descargar desde la interfaz.

La configuración local usa CUDA para generar la voz y CPU para codificar/decodificar el audio, reduciendo el consumo de los 4 GB de memoria de la RTX 2050. La transcripción automática o explícita usa Whisper Base en CPU y requiere descargar ese modelo la primera vez.

Voice Changer usa Seed-VC en `.venv-vc/`. Durante una conversión, MAGI Eleven pasa temporalmente a la CPU para dejar la RTX disponible y vuelve a CUDA al terminar. Seed-VC utiliza características acústicas internas, pero no crea una transcripción ni genera la salida a partir de texto.

Para usar solamente CPU, ejecuta `Iniciar-MAGI-Eleven.cmd -Device cpu` cuando no haya otra instancia abierta.

Archivos:

- `app/`: código base de OmniVoice, commit `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`, adaptado para MAGI Eleven.
- `.venv/`: entorno Python aislado.
- `.venv-vc/`: entorno aislado de Seed-VC para conversión directa.
- `vendor/seed-vc/`: motor local de Voice Changer.
- `models/OmniVoice/`: modelos descargados y verificados con SHA256.
- `logs/server.log` y `logs/server.err`: registros de ejecución.
- `installed-versions.txt`: versiones exactas instaladas.

El servidor escucha únicamente en este equipo. Cerrar la pestaña del navegador no detiene el proceso. Para cerrarlo y liberar memoria, abre **Detener-MAGI-Eleven.cmd**.

Instalación comprobada con `pip check`, respuesta HTTP 200 de la interfaz y generación de `outputs/prueba.wav` (2,9 segundos, 24 kHz) usando CUDA.
