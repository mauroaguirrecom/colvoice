# ColVoice: instrucciones compactas para agentes

Lee este archivo antes de explorar el repositorio. Evita leer `app/omnivoice/cli/demo.py` completo.

## Usar ColVoice

1. Inicia la app con `Iniciar-ColVoice.cmd`.
2. Comprueba la conexión: `.\.venv\Scripts\python.exe colvoice_cli.py health`.
3. Consulta voces: `.\.venv\Scripts\python.exe colvoice_cli.py voices list`.
4. Genera audio con `colvoice_cli.py tts --voice "NOMBRE" --text "..." --output outputs\agente.wav`.
5. Evalúa la pronunciación con `colvoice_cli.py evaluate --audio outputs\agente.wav --text "..."`.
6. Consulta [API.md](API.md) para el contrato JSON.

## Mapa mínimo

- `app/omnivoice/cli/demo.py`: UI Gradio y lógica de generación.
- `colvoice_api.py`: API local compartida con el modelo cargado.
- `colvoice_cli.py`: cliente CLI para agentes.
- `voices/`: biblioteca privada local; no publicarla.
- `outputs/`: resultados generados; no publicarlos.
- `INSTALACION.md`: instalación y uso general.

## Reglas de cambios

- Mantén el idioma predeterminado en español colombiano.
- Conserva `omnivoice` en imports y rutas técnicas por compatibilidad upstream.
- No incluyas modelos, audios personales, voces guardadas, logs ni credenciales en Git.
- Para probar la generación, usa primero la API/CLI; no escanees archivos completos sin necesidad.
