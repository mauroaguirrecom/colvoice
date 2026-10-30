# Instalación y uso de ColVoice

ColVoice es una interfaz local para generar voz, clonar referencias y transformar grabaciones. Está optimizada para español colombiano y funciona también con español mexicano y latinoamericano. La aplicación se ejecuta en Windows, guarda las voces en el equipo y usa el motor OmniVoice como dependencia técnica.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Python 3.10 o superior.
- NVIDIA GPU con CUDA para la ruta recomendada. También puedes usar CPU con tiempos de generación mayores.
- Espacio libre para el entorno Python, la caché de Whisper y los pesos del modelo.

La interfaz de ColVoice se valida principalmente con Windows y NVIDIA. Apple Silicon e Intel XPU son posibilidades del motor upstream, pero no forman parte de la ruta local comprobada de este repositorio.

## Instalación recomendada en Windows

Abre PowerShell en la carpeta del repositorio y crea un entorno limpio:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Para NVIDIA con CUDA 12.8, instala PyTorch y las dependencias de ColVoice:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

### Alternativa con uv

Si prefieres `uv`, crea el entorno y usa el mismo conjunto de dependencias:

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-local.txt
```

En ambos métodos el repositorio se descarga como ColVoice; `omnivoice` se conserva solo como nombre técnico del motor compatible.

Descarga los pesos oficiales en la carpeta local `models/`:

```powershell
.\.venv\Scripts\python.exe download_models.py
```

El modelo no se guarda en Git porque sus archivos son grandes y tienen condiciones de uso propias. El script descarga desde la [fuente oficial de OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

Inicia la aplicación:

```powershell
.\Iniciar-ColVoice.cmd
```

Después abre `http://127.0.0.1:8001`. Para detenerla usa `Detener-ColVoice.cmd`.

### CPU

Si no tienes NVIDIA, instala las ruedas normales de PyTorch antes de instalar el resto:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\Iniciar-ColVoice.cmd -Device cpu
```

La generación en CPU es compatible, pero tarda más durante la carga y la síntesis.

## Inicio rápido

1. Abre **Texto a voz** y escribe el guion.
2. Mantén **Español** como idioma inicial o elige otra variante disponible.
3. Para clonar una voz, carga una referencia limpia de 3 a 10 segundos.
4. Deja **Texto pronunciado** vacío si quieres que Whisper lo transcriba automáticamente, siguiendo el comportamiento de la API oficial.
5. Pulsa **Generar voz** y descarga el resultado desde el reproductor.

Las referencias largas también se aceptan: ColVoice conserva el archivo en la biblioteca y prepara un fragmento limpio para el prompt de clonación. Una referencia corta, clara y sin música suele ser más estable que una grabación muy larga.

## API de clonación

El paquete Python interno conserva el nombre `omnivoice` para mantener compatibilidad con el motor original. Un ejemplo avanzado es:

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

# ref_text es opcional: al omitirlo, el modelo usa Whisper para transcribir ref_audio.
audio = model.generate(
    text="Hola, esta es una prueba de ColVoice.",
    language="es",
    ref_audio="referencia.wav",
)
sf.write("salida.wav", audio[0], 24000)
```

Puedes guardar un prompt de voz y reutilizarlo sin volver a transcribir el audio:

```python
prompt = model.create_voice_clone_prompt(ref_audio="referencia.wav")
prompt.save("mi-voz.pt")

from omnivoice import VoiceClonePrompt
prompt = VoiceClonePrompt.load("mi-voz.pt")
audio = model.generate(text="Un nuevo guion.", language="es", voice_clone_prompt=prompt)
```

## Diseño de voz y voz automática

```python
audio = model.generate(
    text="Una narración breve para un podcast.",
    language="es",
    instruct="female, warm, clear, medium pitch",
)
```

Si no envías referencia ni instrucciones, el motor puede escoger una voz automáticamente:

```python
audio = model.generate(text="Esta frase usa una voz automática.", language="es")
```

La clonación suele ser el modo más consistente. Los perfiles descriptivos pueden variar según el idioma, el texto y los atributos solicitados.

## Control de generación

```python
audio = model.generate(
    text="Texto de ejemplo.",
    language="es",
    num_step=32,  # 16 es más rápido; 32 suele dar mayor estabilidad
    speed=1.0,    # mayor que 1 acelera; menor que 1 ralentiza
    duration=5.0, # duración fija; tiene prioridad sobre speed
)
```

Las etiquetas no verbales y las correcciones fonéticas pertenecen al motor upstream. Consulta su [documentación oficial](https://github.com/k2-fsa/OmniVoice) antes de usarlas en producción.

## Herramientas relacionadas

- [README principal](README.md): resumen del producto y funciones.
- [Voice Changer](VOICE-CHANGER.md): conversión directa opcional con Seed-VC.
- [Créditos y licencias](CREDITS.md): atribución, pesos y condiciones de redistribución.
- [Repositorio upstream OmniVoice](https://github.com/k2-fsa/OmniVoice): API, entrenamiento y evaluación del motor.

No publiques audios personales, prompts de voz, credenciales, pesos descargados ni salidas generadas junto con el código.
