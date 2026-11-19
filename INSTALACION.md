# Instalación y uso de ColVoice

ColVoice es una interfaz local para generar voz, clonar referencias y transformar grabaciones. Está optimizada para español colombiano y funciona también con español mexicano y latinoamericano. Es compatible con Windows, Linux y macOS; guarda las voces en el equipo y usa el motor OmniVoice como dependencia técnica.

## Requisitos

- Windows, Linux o macOS de 64 bits.
- Python 3.10 o superior.
- GPU NVIDIA con CUDA para la ruta de mayor velocidad. También puedes usar CPU o Apple Silicon/MPS con tiempos de generación mayores.
- Espacio libre para el entorno Python, la caché de Whisper y los pesos del modelo.

La interfaz comparte el mismo flujo en los tres sistemas. El instalador selecciona PyTorch CUDA cuando detecta NVIDIA y usa las ruedas portables en CPU o Apple Silicon.

## Instalación recomendada

Desde la carpeta del repositorio ejecuta:

```bash
python setup_colvoice.py
```

El instalador crea `.venv`, instala PyTorch y el resto de dependencias, y descarga el modelo en `models/OmniVoice`. Puedes seleccionar el backend explícitamente:

```bash
python setup_colvoice.py --cuda
python setup_colvoice.py --cpu
python setup_colvoice.py --skip-model
```

### Alternativa con uv

Si prefieres `uv`, crea el entorno y usa el mismo conjunto de dependencias:

```bash
uv venv .venv --python 3.10
uv pip install torch torchaudio
uv pip install -r requirements-local.txt
```

En ambos métodos el repositorio se descarga como ColVoice; `omnivoice` se conserva solo como nombre técnico del motor compatible.

Descarga los pesos oficiales en la carpeta local `models/`:

```bash
python download_models.py
```

El modelo no se guarda en Git porque sus archivos son grandes y tienen condiciones de uso propias. El script descarga desde la [fuente oficial de OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

Inicia la aplicación con el launcher incluido en el repositorio y abre la dirección local que indique el proceso. Para integraciones automatizadas, comprueba primero:

```bash
python colvoice_cli.py health
```

### CPU y Apple Silicon

Usa `python setup_colvoice.py --cpu`. La generación es compatible, aunque tarda más durante la carga y la síntesis que la ruta NVIDIA.

## Instalación con agentes

ChatGPT Codex, Claude Code y Antigravity pueden instalar ColVoice sin explorar el repositorio completo. Pídeles que lean primero [AGENT-SETUP.md](AGENT-SETUP.md), detecten el sistema y la GPU, ejecuten el instalador, validen la API y generen un WAV de prueba. El modelo puede ocupar varios gigabytes: el agente debe solicitar confirmación antes de descargarlo.

## Inicio rápido

1. Abre **Texto a voz** y escribe el guion.
2. Mantén **Español** como idioma inicial o elige otra variante disponible.
3. Para clonar una voz, carga una referencia limpia de 3 a 10 segundos.
4. Deja **Texto pronunciado** vacío si quieres que Whisper lo transcriba automáticamente, siguiendo el comportamiento de la API oficial.
5. Pulsa **Generar voz** y descarga el resultado desde el reproductor.

Las referencias largas también se aceptan: ColVoice conserva el archivo en la biblioteca y prepara un fragmento limpio para el prompt de clonación. Una referencia corta, clara y sin música suele ser más estable que una grabación muy larga.

ColVoice activa una mejora de pronunciación para español: expande abreviaturas frecuentes, porcentajes y números antes de sintetizar. `num2words` se instala automáticamente con `requirements-local.txt`; para palabras o nombres propios muy específicos, escribe la forma fonética deseada directamente en el guion.

## Exportar e importar voces guardadas

Para mover tu biblioteca entre dos instalaciones de ColVoice:

1. Abre **Mis voces**.
2. Pulsa **Exportar biblioteca (.zip)** y guarda el archivo.
3. En la otra instalación, abre **Mis voces** y selecciona el ZIP en **Archivo de biblioteca ColVoice**.
4. Pulsa **Importar biblioteca**.

El ZIP contiene los prompts de clonación, las referencias de audio y los nombres de las voces. Las voces existentes nunca se sobrescriben: si hay un nombre repetido, ColVoice lo importa como `Nombre (importada 2)`.

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

### Evaluar la calidad de una voz

Desde la app abre **Evaluador de calidad**, sube el WAV generado y pega el guion
esperado. El informe muestra precisión estimada, WER y las palabras omitidas o
sustituidas; esas palabras pueden corregirse en el diccionario de pronunciación de
**Mis voces**. También está disponible para agentes mediante `POST /v1/evaluate`.

### Diálogos, similitud y limpieza

- **Modo diálogo:** escribe turnos como `Narradora: Hola` y `Narrador: Bienvenido`, usando voces guardadas en **Mis voces**.
- **Similitud de clonación:** controla el equilibrio entre conservar la identidad de la referencia y permitir una interpretación más natural.
- **Aislamiento y limpieza:** abre **Limpieza de audio** para reducir ruido constante antes de clonar una voz. El resultado se descarga como WAV y puede volver a usarse como referencia.
- **Streaming local:** en **Texto a voz** activa “Generar por fragmentos...” para escuchar avances mientras se procesa el guion completo.

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
- [API para agentes](API.md): endpoints JSON compactos y comandos CLI para Codex, Claude Code y Antigravity.
- [AGENTS.md](AGENTS.md): contexto operativo breve para evitar leer todo el código.
- [Voice Changer](VOICE-CHANGER.md): conversión directa opcional con Seed-VC.
- [Créditos y licencias](CREDITS.md): atribución, pesos y condiciones de redistribución.
- [Repositorio upstream OmniVoice](https://github.com/k2-fsa/OmniVoice): API, entrenamiento y evaluación del motor.

No publiques audios personales, prompts de voz, credenciales, pesos descargados ni salidas generadas junto con el código.
