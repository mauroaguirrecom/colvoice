# ColVoice API para agentes

ColVoice inicia una API local en `http://127.0.0.1:8002`. El servidor comparte el modelo ya cargado por la interfaz, así que un agente no necesita importar Gradio ni leer `app/omnivoice/cli/demo.py`.

## Salud

```bash
curl http://127.0.0.1:8002/v1/health
```

Respuesta:

```json
{"brand":"ColVoice","status":"ok","api_version":"1","model_loaded":true}
```

## Voces guardadas

```bash
curl http://127.0.0.1:8002/v1/voices
```

## Texto a voz

```bash
curl -X POST http://127.0.0.1:8002/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hola desde un agente","voice":"Narradora principal","language":"Spanish","num_step":32,"speed":1.0}'
```

El resultado devuelve la ruta local del WAV generado. `voice` es opcional; si se omite, el motor genera con voz automática. Si la voz tiene un diccionario de pronunciación, se aplica automáticamente.

## CLI recomendado

```bash
python colvoice_cli.py health
python colvoice_cli.py voices list
python colvoice_cli.py tts --voice "Narradora principal" --text "Hola desde Codex" --output outputs/agente.wav
```

La API solo escucha en `127.0.0.1`; no incluye autenticación ni debe exponerse a Internet sin añadir una capa de seguridad.

## Evaluar calidad de un audio

El evaluador usa el ASR local para comparar el audio con el guion esperado. Devuelve
precisión estimada, WER, omisiones, sustituciones, palabras inesperadas y salud básica
de la señal (pico, RMS, silencio y clipping).

```bash
curl -X POST http://127.0.0.1:8002/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"audio_file":"outputs/agente.wav","expected_text":"Hola desde Codex"}'
```

También está disponible desde la CLI:

```bash
python colvoice_cli.py evaluate --audio outputs/agente.wav --text "Hola desde Codex"
```
