# ColVoice: instrucciones compactas para agentes

Lee este archivo antes de explorar el repositorio. Evita leer `app/omnivoice/cli/demo.py` completo.

## Usar ColVoice

1. Si falta `.venv`, ejecuta `python setup_colvoice.py` y solicita confirmación antes de descargar los pesos.
2. Inicia la app con `python run_omnivoice.py`.
3. Comprueba la conexión: `python colvoice_cli.py health`.
4. Consulta voces: `python colvoice_cli.py voices list`.
5. Genera audio con `python colvoice_cli.py tts --voice "NOMBRE" --text "..." --output outputs/agente.wav`.
6. Evalúa la pronunciación con `python colvoice_cli.py evaluate --audio outputs/agente.wav --text "..."`.
7. Consulta [API.md](API.md) para el contrato JSON.

## Mapa mínimo

- `app/omnivoice/cli/demo.py`: UI Gradio y lógica de generación.
- `colvoice_api.py`: API local compartida con el modelo cargado.
- `colvoice_cli.py`: cliente CLI para agentes.
- `voices/`: biblioteca privada local; no publicarla.
- `outputs/`: resultados generados; no publicarlos.
- `INSTALACION.md`: instalación y uso general.
- `AGENT-SETUP.md`: flujo detallado para Codex, Claude Code y Antigravity.

## Reglas de cambios

- Mantén el idioma predeterminado en español colombiano.
- Conserva `omnivoice` en imports y rutas técnicas por compatibilidad upstream.
- No incluyas modelos, audios personales, voces guardadas, logs ni credenciales en Git.
- Para probar la generación, usa primero la API/CLI; no escanees archivos completos sin necesidad.
