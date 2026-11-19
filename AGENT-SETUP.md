# Instalación de ColVoice con agentes

Este archivo es el punto de entrada para ChatGPT Codex, Claude Code, Antigravity
y otros agentes de programación. Permite operar ColVoice sin leer todo
`app/omnivoice/cli/demo.py`.

## Prompt recomendado

Copia este prompt en tu agente:

```text
Clona https://github.com/mauroaguirrecom/colvoice.
Lee primero AGENT-SETUP.md, AGENTS.md y API.md; no leas demo.py completo.
Detecta el sistema operativo y si hay GPU NVIDIA.
Si falta .venv, ejecuta `python setup_colvoice.py --cuda` para NVIDIA o
`python setup_colvoice.py --cpu` para CPU/Apple Silicon.
Antes de descargar los pesos del modelo, pide confirmación porque ocupan varios GB.
Después ejecuta `python colvoice_cli.py health`, lista las voces y genera un WAV de prueba.
No publiques models/, voices/, outputs/, logs/ ni credenciales.
```

## Flujo operativo

```bash
python setup_colvoice.py --skip-model
python setup_colvoice.py --cuda       # NVIDIA
python setup_colvoice.py --cpu        # CPU o Apple Silicon/MPS
python colvoice_cli.py health
python colvoice_cli.py voices list
python colvoice_cli.py tts --text "Hola desde ColVoice" --output outputs/agente.wav
python colvoice_cli.py evaluate --audio outputs/agente.wav --text "Hola desde ColVoice"
```

El instalador selecciona automáticamente CUDA cuando encuentra `nvidia-smi`; si
no lo encuentra, instala el backend portable. El mismo repositorio funciona en
Windows, Linux y macOS. El agente debe usar el intérprete del entorno virtual y
no asumir rutas específicas de un sistema.

## Reglas para agentes

- Usa `colvoice_cli.py` o los endpoints de `API.md`; no escanees el código completo.
- Comprueba `/v1/health` antes de generar.
- Usa nombres exactos de voces obtenidos desde `voices list`.
- Guarda resultados en `outputs/`, que está excluido de Git.
- No subas voces, audios personales, modelos, cachés ni claves.
- Si la API no responde, informa el error y revisa los logs sin borrar datos.
