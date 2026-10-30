#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import html
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict
import uuid

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.utils.audio import load_waveform, trim_long_audio
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
_DEFAULT_LANGUAGE = "Spanish"
_BRAND_NAME = "ColVoice"
_BRAND_TITLE = "ColVoice — Voz Colombiana"


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

_COLOMBIA_PRESETS = {
    "Colombia · Podcast masculino": {
        "language": "Spanish",
        "attributes": {
            "Gender / 性别": "Male / 男",
            "Age / 年龄": "Young Adult / 青年",
            "Pitch / 音调": "Moderate Pitch / 中音调",
        },
        "speed": 0.95,
        "steps": 32,
        "guidance_scale": 2.0,
    },
    "Colombia · Podcast femenino": {
        "language": "Spanish",
        "attributes": {
            "Gender / 性别": "Female / 女",
            "Age / 年龄": "Young Adult / 青年",
            "Pitch / 音调": "Moderate Pitch / 中音调",
        },
        "speed": 0.95,
        "steps": 32,
        "guidance_scale": 2.0,
    },
    "Colombia · Noticias masculino": {
        "language": "Spanish",
        "attributes": {
            "Gender / 性别": "Male / 男",
            "Age / 年龄": "Middle-aged / 中年",
            "Pitch / 音调": "Low Pitch / 低音调",
        },
        "speed": 1.0,
        "steps": 36,
        "guidance_scale": 2.1,
    },
    "Colombia · Noticias femenino": {
        "language": "Spanish",
        "attributes": {
            "Gender / 性别": "Female / 女",
            "Age / 年龄": "Middle-aged / 中年",
            "Pitch / 音调": "Moderate Pitch / 中音调",
        },
        "speed": 1.0,
        "steps": 36,
        "guidance_scale": 2.1,
    },
    "Personalizado": {
        "language": _DEFAULT_LANGUAGE,
        "attributes": {},
        "speed": 1.0,
        "steps": 32,
        "guidance_scale": 2.0,
    },
}
_DEFAULT_COLOMBIA_PRESET = "Colombia · Podcast masculino"

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Explicit source-audio transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    voice_library_dir=None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate
    voice_library = Path(voice_library_dir or (Path.cwd() / "voices")).resolve()
    voice_library.mkdir(parents=True, exist_ok=True)
    voice_index_path = voice_library / "index.json"
    no_saved_voice = "Subir nueva referencia"

    def _read_voice_index():
        if not voice_index_path.is_file():
            return []
        try:
            data = json.loads(voice_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logging.warning("Could not read saved voice index: %s", e)
            return []

        entries = []
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            filename = str(entry.get("file", "")).strip()
            if not name or Path(filename).name != filename:
                continue
            if (voice_library / filename).is_file():
                normalized_entry = {"name": name, "file": filename}
                reference_filename = str(entry.get("reference_audio", "")).strip()
                if (
                    reference_filename
                    and Path(reference_filename).name == reference_filename
                    and (voice_library / reference_filename).is_file()
                ):
                    normalized_entry["reference_audio"] = reference_filename
                if "reference_text_manual" in entry:
                    normalized_entry["reference_text_manual"] = bool(
                        entry["reference_text_manual"]
                    )
                entries.append(normalized_entry)
        return sorted(entries, key=lambda item: item["name"].casefold())

    def _write_voice_index(entries):
        temporary_path = voice_index_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(voice_index_path)

    def _saved_voice_choices():
        return [no_saved_voice] + [entry["name"] for entry in _read_voice_index()]

    def _load_saved_voice(name):
        if not name or name == no_saved_voice:
            return None
        for entry in _read_voice_index():
            if entry["name"] == name:
                reference_filename = entry.get("reference_audio")
                manual_text_flag = entry.get("reference_text_manual")
                if reference_filename:
                    reference_info = sf.info(str(voice_library / reference_filename))
                    if reference_info.duration > 20 and manual_text_flag is True:
                        raise ValueError(
                            f"La voz '{name}' usa una referencia de "
                            f"{reference_info.duration:.1f} segundos. "
                            "Su texto manual no puede alinearse después de recortarla. "
                            "Deja el texto vacío para que ColVoice lo transcriba automáticamente."
                        )
                    reference_path = str(voice_library / reference_filename)
                    # Older profiles may contain the complete long recording in
                    # their prompt. Rebuild those prompts through OmniVoice's
                    # official optional-ref_text path so it trims and transcribes
                    # a clean segment before tokenization.
                    if reference_info.duration > 20 and manual_text_flag is not True:
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=reference_path,
                            ref_text=None,
                        )
                        repaired_path = voice_library / f".{entry['file']}.repair"
                        prompt.save(str(repaired_path))
                        repaired_path.replace(voice_library / entry["file"])
                        entries = _read_voice_index()
                        for saved_entry in entries:
                            if saved_entry["name"] == entry["name"]:
                                saved_entry["reference_text_manual"] = False
                        _write_voice_index(entries)
                        return prompt
                prompt = VoiceClonePrompt.load(str(voice_library / entry["file"]))
                if reference_filename:
                    try:
                        _validate_reference_alignment(
                            prompt,
                            prompt.ref_text,
                            reference_path,
                        )
                    except ValueError:
                        if manual_text_flag is not None:
                            raise
                        # Older profiles did not record whether the text was
                        # typed or ASR-generated. Repair those profiles using
                        # OmniVoice's official optional-ref_text behavior.
                        repaired_text = _transcribe_audio_file(reference_path)
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=reference_path,
                            ref_text=repaired_text,
                        )
                        repaired_path = voice_library / f".{entry['file']}.repair"
                        prompt.save(str(repaired_path))
                        repaired_path.replace(voice_library / entry["file"])
                        entries = _read_voice_index()
                        for saved_entry in entries:
                            if saved_entry["name"] == entry["name"]:
                                saved_entry["reference_text_manual"] = False
                        _write_voice_index(entries)
                return prompt
        raise ValueError(f"La voz guardada '{name}' ya no está disponible.")

    def _load_saved_voice_audio(name):
        if not name or name == no_saved_voice:
            return None
        for entry in _read_voice_index():
            if entry["name"] != name:
                continue
            reference_filename = entry.get("reference_audio")
            if not reference_filename:
                raise ValueError(
                    f"La voz '{name}' se guardó antes de habilitar Voice Changer y "
                    "no conserva su audio de referencia. Carga una referencia nueva."
                )
            return str(voice_library / reference_filename)
        raise ValueError(f"La voz guardada '{name}' ya no está disponible.")

    def _validate_audio_file(audio_path, label="audio"):
        """Return a usable local audio path or raise a clear UI-facing error."""
        if not audio_path:
            raise ValueError(f"Selecciona el {label}.")

        candidate = getattr(audio_path, "path", audio_path)
        path = Path(str(candidate))
        if not path.is_file():
            raise ValueError(
                f"El {label} ya no está disponible. Vuelve a cargar el archivo."
            )
        if path.stat().st_size <= 0:
            raise ValueError(
                f"El {label} está vacío (0 bytes). Vuelve a cargar un audio válido."
            )
        return str(path)

    def _validate_reference_audio_file(audio_path):
        """Validate an uploaded reference used for text-based cloning."""
        path = _validate_audio_file(audio_path, "audio de referencia")
        info = sf.info(path)
        if info.duration < 3:
            raise ValueError(
                "La referencia debe durar al menos 3 segundos para clonar la voz."
            )
        return path

    def _validate_reference_alignment(prompt, transcript, ref_audio):
        """Reject prompts whose text is clearly unrelated to the audio length."""
        text_weight = model.duration_estimator.calculate_total_weight(transcript)
        token_count = int(prompt.ref_audio_tokens.size(-1))
        audio_duration = sf.info(ref_audio).duration
        if text_weight <= 0:
            raise ValueError("El texto de la referencia debe contener palabras.")

        # A normal Spanish sample is roughly 1–8 audio tokens per weighted
        # character. A much larger ratio usually means the user entered a
        # voice name or a short label instead of the words in the recording.
        token_to_text_ratio = token_count / text_weight
        if audio_duration >= 5 and token_to_text_ratio > 18:
            raise ValueError(
                "El texto no parece corresponder al audio de referencia. "
                "Escribe exactamente todas las palabras que se pronuncian "
                "en la muestra y vuelve a guardarla."
            )

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_clone_prompt=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        if not language:
            lang = _DEFAULT_LANGUAGE
        elif language == "Auto":
            lang = None
        else:
            lang = language

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        try:
            if mode == "clone":
                if voice_clone_prompt is not None:
                    kw["voice_clone_prompt"] = voice_clone_prompt
                elif not ref_audio:
                    return None, (
                        "Selecciona una voz guardada o carga un audio de referencia."
                    )
                else:
                    valid_ref_audio = _validate_reference_audio_file(ref_audio)
                    reference_text = (ref_text or "").strip() or None
                    if reference_text and sf.info(valid_ref_audio).duration > 20:
                        raise ValueError(
                            "La referencia supera 20 segundos y el texto manual ya no "
                            "quedaría alineado al recorte automático. Déjalo vacío para "
                            "que ColVoice lo transcriba, o usa un clip corto."
                        )
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=valid_ref_audio,
                        ref_text=reference_text,
                    )
                    if reference_text:
                        _validate_reference_alignment(
                            prompt,
                            reference_text,
                            valid_ref_audio,
                        )
                    kw["voice_clone_prompt"] = prompt

            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    def _transcribe_audio_file(audio_path):
        """Transcribe an uploaded file with the small CPU Whisper model."""
        audio_path = _validate_audio_file(audio_path, "audio para transcribir")
        if model._asr_pipe is None:
            model.load_asr_model(device="cpu")

        waveform, sample_rate = load_waveform(audio_path)
        if waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0, keepdims=True)
        audio_input = {
            "array": np.squeeze(waveform),
            "sampling_rate": sample_rate,
        }
        try:
            result = model._asr_pipe(
                audio_input,
                chunk_length_s=30,
                stride_length_s=5,
            )
            return result["text"].strip()
        except TypeError:
            return model.transcribe((waveform, sample_rate))

    def _audio_duration_seconds(audio_path):
        audio_path = _validate_audio_file(audio_path, "audio original")
        waveform, sample_rate = load_waveform(audio_path)
        return float(waveform.shape[-1]) / float(sample_rate)

    def _save_voice_to_library(name, ref_audio, ref_text):
        display_name = (name or "").strip()
        if not display_name:
            raise ValueError("Escribe un nombre para la voz.")
        if len(display_name) > 60:
            raise ValueError("El nombre de la voz no puede superar 60 caracteres.")
        ref_audio = _validate_reference_audio_file(ref_audio)

        entries = _read_voice_index()
        if any(
            entry["name"].casefold() == display_name.casefold()
            for entry in entries
        ):
            raise ValueError(f"Ya existe una voz llamada '{display_name}'.")

        transcript = (ref_text or "").strip()
        manual_transcript = bool(transcript)
        if manual_transcript and sf.info(ref_audio).duration > 20:
            raise ValueError(
                "La referencia supera 20 segundos y el texto manual ya no quedaría "
                "alineado al recorte automático. Déjalo vacío para que ColVoice "
                "lo transcriba, o usa un clip corto."
            )
        # Passing None is intentional: OmniVoice trims long recordings and then
        # runs Whisper on the same trimmed segment, keeping the prompt aligned.
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=transcript or None,
        )
        if not manual_transcript:
            transcript = prompt.ref_text
        if manual_transcript:
            _validate_reference_alignment(prompt, transcript, ref_audio)

        voice_id = uuid.uuid4().hex
        filename = f"{voice_id}.pt"
        reference_filename = f"{voice_id}_reference.wav"
        temporary_prompt = voice_library / f".{voice_id}.tmp"
        temporary_reference = voice_library / f".{voice_id}_reference.tmp.wav"
        final_prompt = voice_library / filename
        final_reference = voice_library / reference_filename
        prompt.save(str(temporary_prompt))
        temporary_prompt.replace(final_prompt)

        reference_waveform, reference_sample_rate = load_waveform(ref_audio)
        sf.write(
            str(temporary_reference),
            reference_waveform.T,
            reference_sample_rate,
            subtype="PCM_16",
        )
        temporary_reference.replace(final_reference)

        entries.append(
            {
                "name": display_name,
                "file": filename,
                "reference_audio": reference_filename,
                "reference_text_manual": manual_transcript,
            }
        )
        try:
            _write_voice_index(
                sorted(entries, key=lambda item: item["name"].casefold())
            )
        except Exception:
            final_prompt.unlink(missing_ok=True)
            final_reference.unlink(missing_ok=True)
            raise
        return display_name, transcript

    def _move_speech_model(target_device):
        """Move OmniVoice without moving its CPU audio tokenizer."""
        audio_tokenizer = model.audio_tokenizer
        model.audio_tokenizer = None
        try:
            model.to(target_device)
        finally:
            model.audio_tokenizer = audio_tokenizer

    def _run_direct_voice_conversion(
        source_audio,
        saved_voice,
        reference_audio,
        diffusion_steps,
        similarity,
        progress=gr.Progress(),
    ):
        """Run Seed-VC in a separate environment without text transcription."""
        if not source_audio:
            return None, "Selecciona el audio cuya voz deseas cambiar."

        try:
            source_audio = _validate_audio_file(source_audio, "audio original")
            source_duration = _audio_duration_seconds(source_audio)
            if source_duration > 300:
                raise ValueError("El audio original no puede superar 5 minutos.")

            target_audio = _load_saved_voice_audio(saved_voice)
            if target_audio is None:
                target_audio = _validate_audio_file(
                    reference_audio,
                    "audio de referencia de la voz destino",
                )
            target_waveform, target_sample_rate = load_waveform(target_audio)
            target_duration = float(target_waveform.shape[-1]) / float(target_sample_rate)
            if target_duration < 1:
                raise ValueError("La referencia de voz debe durar al menos 1 segundo.")

            workspace_root = Path.cwd().resolve()
            seed_root = workspace_root / "vendor" / "seed-vc"
            seed_python = workspace_root / ".venv-vc" / "Scripts" / "python.exe"
            inference_script = seed_root / "inference.py"
            if not seed_python.is_file() or not inference_script.is_file():
                raise RuntimeError(
                    "El motor Voice Changer no está instalado completamente."
                )

            job_id = uuid.uuid4().hex
            job_dir = workspace_root / ".cache" / "voice-changer" / "jobs" / job_id
            job_output = job_dir / "generated"
            final_dir = workspace_root / "outputs" / "voice-changer"
            job_output.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)

            if target_duration > 15:
                target_waveform = trim_long_audio(
                    target_waveform,
                    target_sample_rate,
                    max_duration=15.0,
                    trim_threshold=15.0,
                )
                clipped_target = job_dir / "target_reference.wav"
                sf.write(
                    str(clipped_target),
                    target_waveform.T,
                    target_sample_rate,
                    subtype="PCM_16",
                )
                target_audio = str(clipped_target)

            command = [
                str(seed_python),
                str(inference_script),
                "--source",
                source_audio,
                "--target",
                target_audio,
                "--output",
                str(job_output),
                "--diffusion-steps",
                str(int(diffusion_steps or 10)),
                "--length-adjust",
                "1.0",
                "--inference-cfg-rate",
                str(float(similarity if similarity is not None else 0.7)),
                "--f0-condition",
                "False",
                "--fp16",
                "True",
            ]

            environment = os.environ.copy()
            model_cache = workspace_root / ".cache" / "seed-vc"
            model_cache.mkdir(parents=True, exist_ok=True)
            environment.update(
                {
                    "HF_HOME": str(model_cache),
                    "TRANSFORMERS_CACHE": str(model_cache / "transformers"),
                    "OMNIVOICE_VC_CACHE": str(model_cache),
                    "PYTHONUTF8": "1",
                    "OMP_NUM_THREADS": "4",
                }
            )

            original_device = next(model.parameters()).device
            progress(0.05, desc="Liberando memoria de la GPU")
            if original_device.type == "cuda":
                _move_speech_model("cpu")
                torch.cuda.empty_cache()

            try:
                progress(
                    0.15,
                    desc="Convirtiendo la voz sin transcribir el audio",
                )
                completed = subprocess.run(
                    command,
                    cwd=str(seed_root),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    if "out of memory" in detail.casefold():
                        raise RuntimeError(
                            "La RTX se quedó sin memoria durante la conversión. "
                            "Prueba con un audio más corto."
                        )
                    raise RuntimeError(
                        "Seed-VC no pudo convertir el audio. "
                        + (detail[-1200:] if detail else "No entregó detalles.")
                    )

                generated_files = sorted(job_output.glob("vc_*.wav"))
                if not generated_files:
                    raise RuntimeError("Seed-VC terminó sin crear el audio de salida.")
                final_path = final_dir / f"voice-changer-{job_id}.wav"
                shutil.copy2(generated_files[0], final_path)
            finally:
                if original_device.type == "cuda":
                    progress(0.92, desc="Restaurando ColVoice en la GPU")
                    torch.cuda.empty_cache()
                    _move_speech_model(original_device)
                shutil.rmtree(job_dir, ignore_errors=True)

            progress(1.0, desc="Voice Changer completado")
            return str(final_path), (
                "Listo. Se reemplazó el timbre sin transcribir ni volver a "
                "sintetizar el texto. La duración y la interpretación provienen "
                "del audio original."
            )
        except subprocess.TimeoutExpired:
            return None, "La conversión superó el tiempo máximo de 30 minutos."
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    def _voice_library_markup():
        entries = _read_voice_index()
        if not entries:
            return """
            <section class="voice-library-empty">
                <div class="empty-wave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
                <h3>Tu biblioteca empieza aquí</h3>
                <p>Guarda una referencia limpia y úsala después en Texto a voz, Cambiar voz o Voice Changer.</p>
            </section>
            """

        cards = []
        for index, entry in enumerate(entries):
            name = html.escape(entry["name"])
            initials = "".join(part[0] for part in name.split()[:2]).upper() or "V"
            direct_badge = (
                "Voice Changer"
                if entry.get("reference_audio")
                else "Solo texto a voz"
            )
            cards.append(
                f"""
                <article class="voice-card">
                    <div class="voice-avatar tone-{index % 4}">{initials}</div>
                    <div class="voice-card-copy">
                        <strong>{name}</strong>
                        <span>{direct_badge}</span>
                    </div>
                    <span class="voice-ready">Local</span>
                </article>
                """
            )
        return '<section class="voice-library-grid">' + "".join(cards) + "</section>"

    def _preview_saved_voice(name):
        if not name or name == no_saved_voice:
            return None, "Selecciona una voz para escuchar su referencia."
        try:
            path = _load_saved_voice_audio(name)
            return path, f"Referencia local de '{name}'."
        except Exception as e:
            return None, f"No se puede previsualizar: {e}"

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=["IBM Plex Sans", "Inter", "Segoe UI", "sans-serif"],
    ).set(
        body_background_fill="#07090e",
        body_text_color="#f8fafc",
        background_fill_primary="#0e131f",
        background_fill_secondary="#141b2d",
        border_color_primary="rgba(255, 255, 255, 0.08)",
        block_background_fill="#0e131f",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_label_background_fill="transparent",
        block_label_text_color="#94a3b8",
        input_background_fill="#080b12",
        input_border_color="rgba(255, 255, 255, 0.1)",
        button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a855f7 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #9333ea 100%)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#141b2d",
        button_secondary_border_color="rgba(255, 255, 255, 0.1)",
        button_secondary_text_color="#e2e8f0",
    )
    css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root, .dark {
        --ov-bg: #07090e;
        --ov-surface: #0e131f;
        --ov-surface-elevated: #141b2d;
        --ov-surface-hover: #1a233a;
        --ov-border: rgba(255, 255, 255, 0.08);
        --ov-border-focus: #6366f1;
        --ov-text-primary: #f8fafc;
        --ov-text-secondary: #94a3b8;
        --ov-text-muted: #64748b;
        --ov-accent: #6366f1;
        --ov-accent-glow: rgba(99, 102, 241, 0.35);
        --ov-sidebar-width: 240px;
        --ov-player-height: 80px;
        
        --body-background-fill: #07090e !important;
        --body-text-color: #f8fafc !important;
        --background-fill-primary: #0e131f !important;
        --background-fill-secondary: #141b2d !important;
        --border-color-primary: rgba(255, 255, 255, 0.08) !important;
        --block-background-fill: #0e131f !important;
        --block-border-color: rgba(255, 255, 255, 0.08) !important;
        --block-label-background-fill: transparent !important;
        --block-label-text-color: #94a3b8 !important;
        --input-background-fill: #080b12 !important;
        --input-border-color: rgba(255, 255, 255, 0.1) !important;
        --button-primary-background-fill: #6366f1 !important;
        --button-primary-text-color: #ffffff !important;
        --button-secondary-background-fill: #141b2d !important;
        --button-secondary-border-color: rgba(255, 255, 255, 0.1) !important;
        --button-secondary-text-color: #e2e8f0 !important;
        color-scheme: dark !important;
    }

    html, body {
        background: #07090e !important;
        color: #f8fafc !important;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        overflow-x: hidden !important;
        margin: 0;
        padding: 0;
    }

    ::selection {
        background: rgba(99, 102, 241, 0.35);
        color: #ffffff;
    }

    * {
        scrollbar-color: #26334d #07090e;
        scrollbar-width: thin;
    }

    .gradio-container {
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        background: #07090e radial-gradient(at 10% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                    radial-gradient(at 90% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%) !important;
        color: #f8fafc !important;
        min-height: 100vh !important;
    }

    .gradio-container > .main {
        width: 100% !important;
        max-width: none !important;
        padding: 0 !important;
    }

    .gradio-container .sidebar-parent {
        width: 100% !important;
        padding-left: var(--ov-sidebar-width) !important;
    }

    /* ==============================================================
       SIDEBAR
       ============================================================== */
    #studio-sidebar {
        width: var(--ov-sidebar-width) !important;
        left: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        transform: none !important;
        padding: 24px 14px calc(var(--ov-player-height) + 20px) !important;
        background: rgba(10, 14, 24, 0.88) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
        z-index: 25 !important;
    }

    #studio-sidebar > button {
        display: none !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 8px 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .studio-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2px;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0ea5a4 0%, #2563eb 100%);
        box-shadow: 0 0 16px rgba(14, 165, 164, 0.42);
        flex: 0 0 auto;
    }

    .studio-mark strong {
        color: #ffffff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: -0.08em;
        line-height: 1;
    }

    .studio-mark i {
        display: block;
        width: 3px;
        border-radius: 3px;
        background: #ffffff;
        animation: studio-wave 1.4s ease-in-out infinite alternate;
    }
    .studio-mark i:nth-child(1) { height: 10px; animation-delay: 0.1s; }
    .studio-mark i:nth-child(2) { height: 20px; animation-delay: 0.3s; }
    .studio-mark i:nth-child(3) { height: 14px; animation-delay: 0.2s; }
    .studio-mark i:nth-child(4) { height: 8px; animation-delay: 0.4s; }

    @keyframes studio-wave {
        0% { transform: scaleY(0.7); opacity: 0.8; }
        100% { transform: scaleY(1.3); opacity: 1; }
    }

    .studio-name {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .studio-subtitle {
        margin-top: 2px;
        font-size: 11px;
        font-weight: 600;
        color: #818cf8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .sidebar-label {
        margin: 24px 10px 10px;
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    #studio-nav {
        border: 0 !important;
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
        box-shadow: none !important;
    }

    #studio-nav [data-testid="block-info"] {
        display: none !important;
    }

    #studio-nav .wrap {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        width: 100% !important;
    }

    #studio-nav label {
        display: flex !important;
        align-items: center !important;
        min-height: 44px !important;
        padding: 0 14px !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        background: transparent !important;
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 180ms cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-sizing: border-box !important;
    }

    #studio-nav label:nth-child(1)::before { content: "🎙️"; font-size: 15px; margin-right: 10px; }
    #studio-nav label:nth-child(2)::before { content: "🔄"; font-size: 15px; margin-right: 10px; }
    #studio-nav label:nth-child(3)::before { content: "⚡"; font-size: 15px; margin-right: 10px; }
    #studio-nav label:nth-child(4)::before { content: "🎨"; font-size: 15px; margin-right: 10px; }
    #studio-nav label:nth-child(5)::before { content: "📁"; font-size: 15px; margin-right: 10px; }

    #studio-nav label:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f1f5f9 !important;
        transform: translateX(3px) !important;
    }

    #studio-nav label:has(input:checked) {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.22) 0%, rgba(139, 92, 246, 0.08) 100%) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        color: #ffffff !important;
        box-shadow: inset 3px 0 0 #6366f1, 0 4px 16px rgba(99, 102, 241, 0.15) !important;
    }

    #studio-nav input {
        display: none !important;
    }

    .sidebar-library {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 24px 6px 0;
        padding: 12px 14px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        font-size: 12px;
        font-weight: 500;
    }

    .sidebar-library strong {
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 700;
    }

    .sidebar-foot {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 14px 6px 0;
        padding: 12px 14px;
        border-radius: 12px;
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .sidebar-foot .status-dot {
        position: relative;
        width: 9px;
        height: 9px;
        flex: 0 0 auto;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
    }

    .sidebar-foot strong {
        display: block;
        font-size: 12px;
        font-weight: 700;
        color: #ecfdf5;
    }

    .sidebar-foot small {
        display: block;
        margin-top: 1px;
        font-size: 11px;
        color: #6ee7b7;
    }

    /* ==============================================================
       MAIN WORKSPACE
       ============================================================== */
    #studio-main {
        width: 100% !important;
        max-width: 1440px !important;
        min-height: 100vh !important;
        margin: 0 auto !important;
        padding: 0 32px calc(var(--ov-player-height) + 36px) !important;
        box-sizing: border-box !important;
    }

    .studio-topbar {
        position: sticky;
        top: 0;
        z-index: 20;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 64px;
        margin: 0 -32px 24px;
        padding: 0 32px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(7, 9, 14, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    .studio-topbar strong {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .studio-topbar span {
        margin-left: 10px;
        color: #64748b;
        font-size: 13px;
        font-weight: 500;
    }

    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
    }

    .privacy-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Workspace panels */
    .studio-view {
        width: 100% !important;
        animation: studio-enter 280ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    @keyframes studio-enter {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .workspace-grid {
        display: grid !important;
        grid-template-columns: minmax(0, 1.45fr) minmax(320px, 1fr) !important;
        gap: 24px !important;
        align-items: start !important;
        width: 100% !important;
    }

    .editor-panel, .workspace-panel, .settings-panel, .result-panel {
        background: rgba(14, 19, 31, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
        overflow: hidden;
    }

    .editor-panel, .workspace-panel {
        padding: 28px !important;
    }

    .settings-panel, .result-panel {
        padding: 24px !important;
    }

    .section-heading {
        margin: 0 0 20px;
    }

    .section-heading h2 {
        margin: 0 0 8px;
        font-size: clamp(24px, 2.2vw, 32px);
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-heading p {
        margin: 0;
        max-width: 65ch;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
    }

    .editor-intro {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .editor-kicker {
        font-size: 12px;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Form controls & inputs */
    .gradio-container .form, .gradio-container .block {
        background: transparent !important;
        border: 0 !important;
    }

    .gradio-container label, .gradio-container .label-wrap, .gradio-container .block-label {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
    }

    .gradio-container [data-testid="block-info"], .field-note {
        color: #64748b !important;
        font-size: 12px !important;
    }

    .gradio-container input, .gradio-container textarea, .gradio-container [role="listbox"] {
        background: #080b12 !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        font-family: inherit !important;
        font-size: 14px !important;
        transition: all 180ms ease !important;
    }

    .editor-panel textarea {
        min-height: 240px !important;
        font-size: 16px !important;
        line-height: 1.65 !important;
    }

    .gradio-container input::placeholder, .gradio-container textarea::placeholder {
        color: #475569 !important;
    }

    .gradio-container input:focus, .gradio-container textarea:focus, .gradio-container [role="listbox"]:focus-visible {
        outline: none !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        background: #0b0f19 !important;
    }

    /* Buttons */
    .gradio-container button {
        border-radius: 12px !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em !important;
        transition: all 180ms cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .gradio-container button.primary.generate-button {
        min-height: 52px !important;
        margin-top: 16px !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a855f7 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        cursor: pointer !important;
    }

    .gradio-container button.primary.generate-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        filter: brightness(1.06);
    }

    .gradio-container button.primary.generate-button:active {
        transform: translateY(1px) scale(0.99) !important;
    }

    .gradio-container button.secondary {
        background: #141b2d !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        font-size: 13px !important;
    }

    .gradio-container button.secondary:hover {
        background: #1e293b !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
    }

    .quick-start-label {
        margin: 18px 0 8px;
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .gradio-container button.secondary.prompt-chip {
        min-height: 32px !important;
        padding: 0 14px !important;
        border-radius: 999px !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #94a3b8 !important;
        font-size: 12px !important;
    }

    .gradio-container button.secondary.prompt-chip:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* Sliders */
    .gradio-container input[type="range"] {
        accent-color: #6366f1 !important;
    }

    /* Accordion */
    .gradio-container .accordion {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background: rgba(8, 12, 20, 0.6) !important;
        overflow: hidden;
    }

    .gradio-container .accordion > button {
        background: rgba(8, 12, 20, 0.6) !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
    }

    /* Audio components */
    .compact-audio {
        border: 1px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 14px !important;
        background: rgba(8, 12, 20, 0.6) !important;
        padding: 8px !important;
        transition: border-color 0.2s ease !important;
    }

    .compact-audio:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
    }

    .output-audio {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        background: rgba(8, 12, 20, 0.7) !important;
        padding: 10px !important;
    }

    .status-field textarea {
        min-height: 64px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        color: #a5b4fc !important;
        background: rgba(8, 12, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Voice Library */
    .voice-library-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }

    .voice-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px;
        border-radius: 14px;
        background: rgba(20, 27, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 180ms ease;
    }

    .voice-card:hover {
        background: rgba(26, 36, 60, 0.8);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    .voice-avatar {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff;
        font-weight: 700;
        font-size: 13px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .voice-avatar.tone-1 { background: linear-gradient(135deg, #ec4899, #f43f5e); }
    .voice-avatar.tone-2 { background: linear-gradient(135deg, #06b6d4, #3b82f6); }
    .voice-avatar.tone-3 { background: linear-gradient(135deg, #10b981, #059669); }

    .voice-card-copy strong {
        display: block;
        color: #f8fafc;
        font-size: 14px;
    }

    .voice-card-copy span {
        display: block;
        margin-top: 2px;
        color: #64748b;
        font-size: 11px;
    }

    .voice-ready {
        color: #10b981;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ==============================================================
       PERSISTENT AUDIO PLAYER (BOTTOM DOCK)
       ============================================================== */
    .persistent-player {
        position: fixed !important;
        z-index: 30;
        left: var(--ov-sidebar-width);
        right: auto !important;
        width: calc(100vw - var(--ov-sidebar-width)) !important;
        box-sizing: border-box !important;
        bottom: 0;
        min-height: var(--ov-player-height);
        padding: 10px 32px !important;
        border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(8, 12, 20, 0.94) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 -12px 34px rgba(0, 0, 0, 0.6) !important;
    }

    .player-layout {
        display: flex !important;
        align-items: center !important;
        gap: 20px !important;
    }

    .player-meta {
        min-width: 160px;
    }

    .player-meta strong {
        display: block;
        color: #f8fafc;
        font-size: 13px;
        font-weight: 700;
    }

    .player-meta span {
        display: block;
        margin-top: 2px;
        color: #818cf8;
        font-size: 11px;
    }

    #footer-audio {
        min-height: 52px !important;
        min-width: 0 !important;
        width: auto !important;
        border: 0 !important;
        background: transparent !important;
    }

    #footer-download {
        min-width: 136px !important;
        flex: 0 0 136px !important;
        margin-left: 4px !important;
    }

    #footer-download button {
        min-height: 40px !important;
        border: 1px solid rgba(129, 140, 248, 0.42) !important;
        border-radius: 10px !important;
        background: rgba(99, 102, 241, 0.14) !important;
        color: #c7d2fe !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }

    #footer-download button:hover {
        background: rgba(99, 102, 241, 0.28) !important;
        border-color: rgba(129, 140, 248, 0.72) !important;
    }

    /* Word Counter Badge (Injected by JS) */
    .ov-stats-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Responsive */
    @media (max-width: 1024px) {
        .workspace-grid {
            grid-template-columns: 1fr !important;
        }
    }

    @media (max-width: 768px) {
        .gradio-container .sidebar-parent {
            padding-left: 0 !important;
        }
        #studio-sidebar {
            display: none !important;
        }
        .persistent-player {
            left: 0 !important;
            width: 100vw !important;
            padding: 8px 16px !important;
        }
        .player-meta {
            display: none !important;
        }
        #studio-main {
            padding: 0 16px calc(var(--ov-player-height) + 24px) !important;
        }
        .studio-topbar {
            margin: 0 -16px 16px;
            padding: 0 16px;
        }
    }
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Idioma", value=_DEFAULT_LANGUAGE):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Español es el idioma predeterminado. Usa Auto solo si quieres detectarlo.",
        )

    # Reusable generation controls. Keep the two creative controls visible and
    # place implementation details behind progressive disclosure.
    def _gen_settings(speed_value=1.0, steps_value=32, guidance_value=2.0):
        sp = gr.Slider(
            0.5,
            1.5,
            value=speed_value,
            step=0.05,
            label="Ritmo",
            info="Lento  ·  Natural  ·  Rápido",
        )
        gs = gr.Slider(
            0.0,
            4.0,
            value=guidance_value,
            step=0.1,
            label="Similitud",
            info="Aumenta la fidelidad a la voz y a la dirección.",
        )
        with gr.Accordion("Más ajustes", open=False):
            du = gr.Number(
                value=None,
                label="Duración fija (segundos)",
                info=(
                    "Déjalo vacío para controlar el ritmo con Velocidad."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=steps_value,
                step=1,
                label="Pasos de inferencia",
                info="32 es el valor recomendado. Menos pasos generan más rápido.",
            )
            dn = gr.Checkbox(
                label="Reducir ruido",
                value=True,
                info="Limpia artefactos durante la generación.",
            )
            pp = gr.Checkbox(
                label="Preparar referencia",
                value=True,
                info="Recorta silencios y prepara el texto de referencia.",
            )
            po = gr.Checkbox(
                label="Limpiar resultado",
                value=True,
                info="Elimina silencios largos del audio generado.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title=_BRAND_TITLE) as demo:
        with gr.Sidebar(
            open=True,
            width=260,
            position="left",
            elem_id="studio-sidebar",
        ):
            gr.HTML(
                """
                <div class="sidebar-brand">
                    <span class="studio-mark" aria-label="Logo ColVoice"><strong>CV</strong><i></i><i></i></span>
                    <div>
                        <div class="studio-name">{_BRAND_NAME}</div>
                        <div class="studio-subtitle">Studio local</div>
                    </div>
                </div>
                <div class="sidebar-label">Herramientas</div>
                """
            )
            studio_nav = gr.Radio(
                choices=[
                    "Texto a voz",
                    "Cambiar voz",
                    "Voice Changer",
                    "Diseñar voz",
                    "Mis voces",
                ],
                value="Texto a voz",
                label="",
                elem_id="studio-nav",
            )
            gr.HTML(
                f"""
                <div class="sidebar-library">
                    <span>Biblioteca de voces</span>
                    <strong>{len(_read_voice_index())}</strong>
                </div>
                <div class="sidebar-foot">
                    <span class="status-dot"></span>
                    <div><strong>RTX activa</strong><small>Procesamiento local</small></div>
                </div>
                """
            )

        with gr.Column(elem_id="studio-main"):
            gr.HTML(
                """
                <header class="studio-topbar">
                    <div>
                        <strong>{_BRAND_TITLE}</strong>
                        <span>Creación y conversión de voz</span>
                    </div>
                    <span class="privacy-badge">100% Local & Privado</span>
                </header>
                <script>
                (function() {
                    // 1. Keyboard shortcut: Ctrl+Enter / Cmd+Enter to Generate
                    document.addEventListener('keydown', function(e) {
                        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                            const activeViews = document.querySelectorAll('.studio-view');
                            for (const view of activeViews) {
                                if (view.offsetParent !== null) {
                                    const btn = view.querySelector('.generate-button button, button.generate-button');
                                    if (btn && !btn.disabled) {
                                        e.preventDefault();
                                        btn.click();
                                        break;
                                    }
                                }
                            }
                        }
                    });

                    // 2. Real-time stats badge on textareas
                    function setupWordCounters() {
                        const textareas = document.querySelectorAll('#studio-main textarea');
                        textareas.forEach(ta => {
                            if (ta.dataset.hasCounter) return;
                            ta.dataset.hasCounter = "true";
                            
                            const badge = document.createElement('div');
                            badge.className = 'ov-stats-badge';
                            badge.textContent = '⚡ 0 palabras · 0 caracteres · ~0s';
                            if (ta.parentNode) {
                                ta.parentNode.appendChild(badge);
                            }

                            function updateStats() {
                                const text = ta.value.trim();
                                const words = text ? text.split(/\\s+/).length : 0;
                                const chars = text.length;
                                const duration = Math.round((words / 140) * 60);
                                badge.textContent = `⚡ ${words} palabras · ${chars} caracteres · ~${duration}s de audio (Ctrl+Enter para generar)`;
                            }
                            ta.addEventListener('input', updateStats);
                            updateStats();
                        });
                    }

                    setInterval(setupWordCounters, 1000);
                })();
                </script>
                """
            )
            with gr.Row(
                elem_id="persistent-player",
                elem_classes=["persistent-player", "player-layout"],
            ):
                gr.HTML(
                    """
                    <div class="player-meta">
                        <strong>Reproductor</strong>
                        <span>Tu última generación permanece aquí</span>
                    </div>
                    """
                )
                footer_audio = gr.Audio(
                    label="",
                    show_label=False,
                    interactive=False,
                    buttons=["download"],
                    type="filepath",
                    elem_id="footer-audio",
                    scale=8,
                )
                footer_download = gr.DownloadButton(
                    "Descargar audio",
                    visible=True,
                    interactive=False,
                    elem_id="footer-download",
                    elem_classes="footer-download",
                    scale=0,
                    min_width=136,
                )

            def _sync_footer_download(audio):
                if not audio:
                    return gr.update(value=None, visible=True, interactive=False)
                if isinstance(audio, dict):
                    path = audio.get("path") or audio.get("url")
                elif isinstance(audio, (tuple, list)):
                    path = (
                        audio[0]
                        if audio and isinstance(audio[0], (str, Path))
                        else None
                    )
                else:
                    path = audio
                if not isinstance(path, (str, Path)):
                    path = None
                return gr.update(
                    value=path,
                    visible=True,
                    interactive=bool(path),
                )

            footer_audio.change(
                _sync_footer_download,
                inputs=footer_audio,
                outputs=footer_download,
            )
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.Group(visible=True, elem_classes="studio-view") as clone_view:
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=7, elem_classes="editor-panel"):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Texto a voz</h2>
                                <p>Escribe el guion, selecciona una voz y ajusta la interpretación. El resultado aparecerá en el reproductor inferior.</p>
                            </div>
                            """
                        )
                        with gr.Row():
                            vc_saved_voice = gr.Dropdown(
                                label="Voz seleccionada",
                                choices=_saved_voice_choices(),
                                value=no_saved_voice,
                                scale=4,
                            )
                        gr.HTML(
                            "<div class='editor-intro'><span class='editor-kicker'>Guion de la narración</span><span class='editor-kicker'>Procesamiento local</span></div>"
                        )
                        vc_text = gr.Textbox(
                            label="",
                            show_label=False,
                            lines=12,
                            placeholder="Escribe o pega aquí el texto que quieres convertir en voz...",
                        )
                        gr.HTML("<div class='quick-start-label'>Empezar con un ejemplo</div>")
                        with gr.Row():
                            vc_example_news = gr.Button(
                                "Noticias",
                                variant="secondary",
                                elem_classes="prompt-chip",
                            )
                            vc_example_podcast = gr.Button(
                                "Podcast",
                                variant="secondary",
                                elem_classes="prompt-chip",
                            )
                            vc_example_intro = gr.Button(
                                "Presentación",
                                variant="secondary",
                                elem_classes="prompt-chip",
                            )
                        with gr.Accordion("Usar o guardar una referencia nueva", open=False):
                            vc_ref_audio = gr.Audio(
                                label="Audio de referencia",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            gr.HTML(
                                "<p class='field-note'>Ideal: 3 a 10 segundos, sin música ni eco. Si cargas una grabación larga, ColVoice recorta un fragmento limpio y lo transcribe automáticamente.</p>"
                            )
                            vc_ref_text = gr.Textbox(
                                label="Texto pronunciado en la referencia (opcional)",
                                lines=2,
                                placeholder="Opcional. Déjalo vacío para transcribirlo automáticamente.",
                            )
                            with gr.Row(elem_classes="voice-save-row"):
                                vc_voice_name = gr.Textbox(
                                    label="Nombre de la voz",
                                    placeholder="Ejemplo: Narrador principal",
                                )
                                vc_save_voice = gr.Button(
                                    "Guardar en mi biblioteca",
                                    variant="secondary",
                                    elem_classes="utility-button",
                                )
                        vc_btn = gr.Button(
                            "Generar audio",
                            variant="primary",
                            elem_classes="generate-button",
                        )
                    with gr.Column(scale=4, elem_classes="settings-panel"):
                        gr.HTML(
                            """
                            <div class="settings-title">
                                <strong>Configuración</strong>
                                <span>Ajusta el idioma, la intención y el comportamiento del modelo.</span>
                            </div>
                            """
                        )
                        vc_lang = _lang_dropdown("Idioma")
                        with gr.Accordion("Dirección de voz", open=True):
                            vc_instruct = gr.Dropdown(
                                label="Perfil de interpretación",
                                choices=[
                                    ("Conservar la referencia", ""),
                                    (
                                        "Narrador masculino natural",
                                        "male, young adult, moderate pitch",
                                    ),
                                    (
                                        "Narradora femenina natural",
                                        "female, young adult, moderate pitch",
                                    ),
                                    ("Voz masculina grave", "male, low pitch"),
                                    ("Voz femenina aguda", "female, high pitch"),
                                    ("Susurro", "whisper"),
                                ],
                                value="",
                                allow_custom_value=False,
                                info=(
                                    "El acento se toma del audio de referencia. "
                                    "Estos perfiles solo ajustan timbre y estilo."
                                ),
                            )
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_audio = gr.Audio(
                            label="Audio generado",
                            type="numpy",
                            buttons=["download"],
                            elem_classes="output-audio",
                            visible=False,
                        )
                        vc_status = gr.Textbox(
                            label="Actividad",
                            lines=3,
                            elem_classes="status-field",
                        )

                def _clone_fn(
                    text,
                    lang,
                    saved_voice,
                    ref_aud,
                    ref_text,
                    instruct,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                ):
                    try:
                        saved_prompt = _load_saved_voice(saved_voice)
                    except Exception as e:
                        return None, f"Error: {type(e).__name__}: {e}"
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                        voice_clone_prompt=saved_prompt,
                    )

                def _save_clone_voice_fn(name, ref_audio, ref_text):
                    try:
                        saved_name, transcript = _save_voice_to_library(
                            name,
                            ref_audio,
                            ref_text,
                        )
                        return (
                            gr.update(
                                choices=_saved_voice_choices(),
                                value=saved_name,
                            ),
                            "",
                            transcript,
                            f"Voz '{saved_name}' guardada localmente.",
                        )
                    except Exception as e:
                        return (
                            gr.update(),
                            name or "",
                            ref_text or "",
                            f"Error al guardar: {type(e).__name__}: {e}",
                        )

                vc_example_news.click(
                    lambda: (
                        "Hoy revisamos una noticia que puede cambiar la forma en "
                        "que las empresas colombianas presentan su información financiera."
                    ),
                    outputs=[vc_text],
                )
                vc_example_podcast.click(
                    lambda: (
                        "Bienvenidos. En este episodio explicaremos, con ejemplos "
                        "claros, el tema contable que está generando más preguntas esta semana."
                    ),
                    outputs=[vc_text],
                )
                vc_example_intro.click(
                    lambda: (
                        f"Hola, soy tu narrador de {_BRAND_TITLE}. Preparé esta "
                        "presentación para acompañarte paso a paso."
                    ),
                    outputs=[vc_text],
                )

                vc_save_voice.click(
                    _save_clone_voice_fn,
                    inputs=[vc_voice_name, vc_ref_audio, vc_ref_text],
                    outputs=[
                        vc_saved_voice,
                        vc_voice_name,
                        vc_ref_text,
                        vc_status,
                    ],
                )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_saved_voice,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[footer_audio, vc_status],
                )

            # ==============================================================
            # Audio to cloned voice
            # ==============================================================
            with gr.Group(visible=False, elem_classes="studio-view") as audio_view:
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=6, elem_classes="workspace-panel"):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Cambia la voz de una grabación</h2>
                                <p>Sube tu narración, revisa la transcripción y genera una nueva versión con otra voz. El contenido hablado se conserva; la música y las pausas exactas no.</p>
                            </div>
                            """
                        )
                        st_source_audio = gr.Audio(
                            label="Audio original",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        st_source_text = gr.Textbox(
                            label="Transcripción del audio (editable)",
                            lines=5,
                            placeholder=(
                                "Puedes escribir o corregir el texto. Si lo "
                                "dejas vacío, Whisper lo transcribe automáticamente."
                            ),
                        )
                        st_transcribe_btn = gr.Button(
                            "Transcribir audio",
                            variant="secondary",
                        )
                        with gr.Row():
                            st_saved_voice = gr.Dropdown(
                                label="Voz para el resultado",
                                choices=_saved_voice_choices(),
                                value=no_saved_voice,
                                scale=4,
                            )
                        st_ref_audio = gr.Audio(
                            label="Referencia de la nueva voz",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.HTML(
                            "<p class='field-note'>Puedes omitirla cuando selecciones una voz guardada.</p>"
                        )
                        with gr.Accordion("Guardar y describir la nueva voz", open=False):
                            st_ref_text = gr.Textbox(
                                label="Texto pronunciado en la referencia (opcional)",
                                lines=2,
                                placeholder=(
                                    "Opcional. Déjalo vacío para transcribirlo automáticamente."
                                ),
                            )
                            with gr.Row(elem_classes="voice-save-row"):
                                st_voice_name = gr.Textbox(
                                    label="Nombre de la voz",
                                    placeholder="Ejemplo: Voz del podcast",
                                )
                                st_save_voice = gr.Button(
                                    "Guardar en mi biblioteca",
                                    variant="secondary",
                                    elem_classes="utility-button",
                                )
                        st_lang = _lang_dropdown(
                            label="Idioma del resultado",
                            value="Spanish",
                        )
                        st_keep_duration = gr.Checkbox(
                            label="Conservar la duración aproximada del original",
                            value=True,
                        )
                        (
                            st_ns,
                            st_gs,
                            st_dn,
                            st_sp,
                            st_du,
                            st_pp,
                            st_po,
                        ) = _gen_settings()
                        st_generate_btn = gr.Button(
                            "Reemplazar voz",
                            variant="primary",
                            elem_classes="generate-button",
                        )
                    with gr.Column(scale=4, elem_classes="result-panel"):
                        gr.HTML(
                            """
                            <div class="result-heading">
                                <h3>Nueva versión</h3>
                                <p>Escucha y descarga el audio una vez termine el proceso.</p>
                            </div>
                            """
                        )
                        st_audio = gr.Audio(
                            label="Audio con la voz clonada",
                            type="numpy",
                            buttons=["download"],
                            elem_classes="output-audio",
                            visible=False,
                        )
                        st_status = gr.Textbox(
                            label="Estado",
                            lines=3,
                            elem_classes="status-field",
                        )

                def _transcribe_source_fn(source_audio):
                    try:
                        transcript = _transcribe_audio_file(source_audio)
                        return transcript, (
                            "Transcripción lista. Puedes corregirla antes de generar."
                        )
                    except Exception as e:
                        return "", f"Error al transcribir: {type(e).__name__}: {e}"

                def _speech_to_speech_fn(
                    source_audio,
                    source_text,
                    ref_audio,
                    ref_text,
                    saved_voice,
                    lang,
                    keep_duration,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                ):
                    if not source_audio:
                        return None, source_text or "", ref_text or "", (
                            "Selecciona el audio original."
                        )
                    try:
                        source_audio = _validate_audio_file(
                            source_audio,
                            "audio original",
                        )
                        saved_prompt = _load_saved_voice(saved_voice)
                        if saved_prompt is None and not ref_audio:
                            return None, source_text or "", ref_text or "", (
                                "Selecciona una voz guardada o carga el audio "
                                "de la voz que deseas clonar."
                            )
                        if saved_prompt is None:
                            ref_audio = _validate_reference_audio_file(ref_audio)

                        transcript = (source_text or "").strip()
                        if not transcript:
                            transcript = _transcribe_audio_file(source_audio)

                        if saved_prompt is not None:
                            reference_transcript = saved_prompt.ref_text
                        else:
                            reference_transcript = (ref_text or "").strip()
                            # Keep this None when the field is blank. The model
                            # then trims long references before Whisper runs,
                            # keeping the transcript aligned with the prompt.
                            if not reference_transcript:
                                reference_transcript = None

                        target_duration = du
                        if keep_duration:
                            target_duration = _audio_duration_seconds(source_audio)

                        audio, status = _gen(
                            transcript,
                            lang,
                            ref_audio,
                            None,
                            ns,
                            gs,
                            dn,
                            sp,
                            target_duration,
                            pp,
                            po,
                            mode="clone",
                            ref_text=reference_transcript,
                            voice_clone_prompt=saved_prompt,
                        )
                        if audio is None:
                            return (
                                None,
                                transcript,
                                reference_transcript or "",
                                status,
                            )

                        duration_note = ""
                        if keep_duration:
                            duration_note = (
                                f" Duración objetivo: {target_duration:.1f} segundos."
                            )
                        return (
                            audio,
                            transcript,
                            reference_transcript or "",
                            "Listo. La voz fue reemplazada." + duration_note,
                        )
                    except Exception as e:
                        return (
                            None,
                            source_text or "",
                            ref_text or "",
                            f"Error: {type(e).__name__}: {e}",
                        )

                def _save_replacement_voice_fn(name, ref_audio, ref_text):
                    try:
                        saved_name, transcript = _save_voice_to_library(
                            name,
                            ref_audio,
                            ref_text,
                        )
                        return (
                            gr.update(
                                choices=_saved_voice_choices(),
                                value=saved_name,
                            ),
                            "",
                            transcript,
                            f"Voz '{saved_name}' guardada localmente.",
                        )
                    except Exception as e:
                        return (
                            gr.update(),
                            name or "",
                            ref_text or "",
                            f"Error al guardar: {type(e).__name__}: {e}",
                        )

                st_transcribe_btn.click(
                    _transcribe_source_fn,
                    inputs=[st_source_audio],
                    outputs=[st_source_text, st_status],
                )

                st_save_voice.click(
                    _save_replacement_voice_fn,
                    inputs=[st_voice_name, st_ref_audio, st_ref_text],
                    outputs=[
                        st_saved_voice,
                        st_voice_name,
                        st_ref_text,
                        st_status,
                    ],
                )

                st_generate_btn.click(
                    _speech_to_speech_fn,
                    inputs=[
                        st_source_audio,
                        st_source_text,
                        st_ref_audio,
                        st_ref_text,
                        st_saved_voice,
                        st_lang,
                        st_keep_duration,
                        st_ns,
                        st_gs,
                        st_dn,
                        st_sp,
                        st_du,
                        st_pp,
                        st_po,
                    ],
                    outputs=[footer_audio, st_source_text, st_ref_text, st_status],
                )

            # ==============================================================
            # ==============================================================
            # Direct voice conversion (Seed-VC)
            # ==============================================================
            with gr.Group(visible=False, elem_classes="studio-view") as changer_view:
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=6, elem_classes="workspace-panel"):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Voice Changer</h2>
                                <p>Reemplaza únicamente el timbre y conserva la actuación, el ritmo, las pausas y la pronunciación del audio original. Este flujo no transcribe ni regenera el texto.</p>
                            </div>
                            """
                        )
                        direct_source_audio = gr.Audio(
                            label="Audio original",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.HTML(
                            "<p class='field-note'>Usa una pista de voz limpia. La música y otros sonidos también pueden sufrir alteraciones.</p>"
                        )
                        with gr.Row():
                            direct_saved_voice = gr.Dropdown(
                                label="Voz para el resultado",
                                choices=_saved_voice_choices(),
                                value=no_saved_voice,
                                scale=4,
                            )
                        direct_reference_audio = gr.Audio(
                            label="Referencia de la voz destino",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.HTML(
                            "<p class='field-note'>Ideal: 3 a 10 segundos. Las grabaciones largas también se aceptan y se recortan para preparar la voz; puedes omitirla cuando la voz guardada conserve su audio original.</p>"
                        )
                        with gr.Accordion("Calidad de conversión", open=False):
                            direct_steps = gr.Slider(
                                4,
                                20,
                                value=10,
                                step=1,
                                label="Pasos de conversión",
                                info="10 equilibra calidad y velocidad en la RTX 2050.",
                            )
                            direct_similarity = gr.Slider(
                                0.0,
                                1.2,
                                value=0.7,
                                step=0.05,
                                label="Fidelidad a la voz destino",
                                info="Valores altos aumentan la semejanza, pero pueden introducir artefactos.",
                            )
                        direct_generate_btn = gr.Button(
                            "Cambiar voz conservando la interpretación",
                            variant="primary",
                            elem_classes="generate-button",
                        )
                    with gr.Column(scale=4, elem_classes="result-panel"):
                        gr.HTML(
                            """
                            <div class="result-heading">
                                <h3>Conversión directa</h3>
                                <p>El resultado mantiene la línea de tiempo del audio original y aplica la identidad de la voz elegida.</p>
                            </div>
                            """
                        )
                        direct_output_audio = gr.Audio(
                            label="Audio con la voz reemplazada",
                            type="filepath",
                            buttons=["download"],
                            elem_classes="output-audio",
                            visible=False,
                        )
                        direct_status = gr.Textbox(
                            label="Estado",
                            lines=4,
                            elem_classes="status-field",
                        )

                direct_generate_btn.click(
                    _run_direct_voice_conversion,
                    inputs=[
                        direct_source_audio,
                        direct_saved_voice,
                        direct_reference_audio,
                        direct_steps,
                        direct_similarity,
                    ],
                    outputs=[footer_audio, direct_status],
                )

            # Voice Design
            # ==============================================================
            with gr.Group(visible=False, elem_classes="studio-view") as design_view:
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=6, elem_classes="workspace-panel"):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Diseña una voz desde cero</h2>
                                <p>Elige un perfil de partida y ajusta edad, tono y ritmo para crear una voz de narración.</p>
                            </div>
                            """
                        )
                        _default_preset = _COLOMBIA_PRESETS[
                            _DEFAULT_COLOMBIA_PRESET
                        ]
                        vd_preset = gr.Dropdown(
                            label="Perfil de partida",
                            choices=list(_COLOMBIA_PRESETS),
                            value=_DEFAULT_COLOMBIA_PRESET,
                            info=(
                                "Configura idioma, edad, tono y ritmo con un clic."
                            ),
                        )
                        vd_text = gr.Textbox(
                            label="Texto de la narración",
                            lines=8,
                            placeholder=(
                                "Escribe el texto que quieres narrar en español..."
                            ),
                        )
                        vd_lang = _lang_dropdown(
                            label="Idioma del resultado",
                            value=_default_preset["language"],
                        )

                        gr.HTML(
                            """
                            <p class="field-note">Los perfiles colombianos ajustan idioma, tono y ritmo. Para conservar un acento específico, usa <strong>Texto a voz</strong> con una referencia real.</p>
                            """
                        )

                        _AUTO = "Auto"
                        vd_groups = []
                        with gr.Accordion("Características de la voz", open=True):
                            for _cat, _choices in _CATEGORIES.items():
                                vd_groups.append(
                                    gr.Dropdown(
                                        label=_cat,
                                        choices=[_AUTO] + _choices,
                                        value=_default_preset["attributes"].get(
                                            _cat, _AUTO
                                        ),
                                        info=_ATTR_INFO.get(_cat),
                                    )
                                )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings(
                            speed_value=_default_preset["speed"],
                            steps_value=_default_preset["steps"],
                            guidance_value=_default_preset["guidance_scale"],
                        )
                        vd_btn = gr.Button(
                            "Generar voz",
                            variant="primary",
                            elem_classes="generate-button",
                        )
                    with gr.Column(scale=4, elem_classes="result-panel"):
                        gr.HTML(
                            """
                            <div class="result-heading">
                                <h3>Vista previa</h3>
                                <p>Prueba el perfil y vuelve a ajustar sus características cuando lo necesites.</p>
                            </div>
                            """
                        )
                        vd_audio = gr.Audio(
                            label="Audio generado",
                            type="numpy",
                            buttons=["download"],
                            elem_classes="output-audio",
                            visible=False,
                        )
                        vd_status = gr.Textbox(
                            label="Estado",
                            lines=3,
                            elem_classes="status-field",
                        )

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                def _apply_voice_preset(preset_name):
                    preset = _COLOMBIA_PRESETS.get(
                        preset_name,
                        _COLOMBIA_PRESETS[_DEFAULT_COLOMBIA_PRESET],
                    )
                    attributes = preset["attributes"]
                    return (
                        preset["language"],
                        *(attributes.get(category, _AUTO) for category in _CATEGORIES),
                        preset["speed"],
                        preset["steps"],
                        preset["guidance_scale"],
                    )

                vd_preset.change(
                    _apply_voice_preset,
                    inputs=[vd_preset],
                    outputs=[vd_lang] + vd_groups + [vd_sp, vd_ns, vd_gs],
                )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[footer_audio, vd_status],
                )

            # ==============================================================
            # Saved voice library
            # ==============================================================
            with gr.Group(visible=False, elem_classes="studio-view") as library_view:
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=7, elem_classes="editor-panel"):
                        gr.HTML(
                            """
                            <div class="section-heading">
                                <h2>Mis voces</h2>
                                <p>Organiza tus referencias locales, escúchalas y reutilízalas en cualquier flujo.</p>
                            </div>
                            """
                        )
                        library_html = gr.HTML(_voice_library_markup())
                        with gr.Row():
                            library_voice = gr.Dropdown(
                                label="Escuchar una voz guardada",
                                choices=_saved_voice_choices(),
                                value=no_saved_voice,
                                scale=4,
                            )
                    with gr.Column(scale=4, elem_classes="settings-panel"):
                        gr.HTML(
                            """
                            <div class="settings-title">
                                <strong>Guardar una voz</strong>
                                <span>La referencia queda en este equipo y estará disponible en todos los generadores.</span>
                            </div>
                            """
                        )
                        library_ref_audio = gr.Audio(
                            label="Referencia limpia",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.HTML(
                            "<p class='field-note'>Puedes cargar una grabación larga; se conserva en tu biblioteca y ColVoice usa un fragmento corto para clonar la voz.</p>"
                        )
                        library_voice_name = gr.Textbox(
                            label="Nombre",
                            placeholder="Ejemplo: Narradora de noticias",
                        )
                        library_ref_text = gr.Textbox(
                            label="Texto pronunciado (opcional)",
                            lines=2,
                            placeholder="Opcional. Déjalo vacío para transcribirlo automáticamente.",
                        )
                        library_save = gr.Button(
                            "Guardar en mi biblioteca",
                            variant="primary",
                            elem_classes="generate-button",
                        )
                        library_status = gr.Textbox(
                            label="Actividad",
                            lines=3,
                            elem_classes="status-field",
                        )

                def _save_library_voice_fn(name, ref_audio, ref_text):
                    try:
                        saved_name, transcript = _save_voice_to_library(
                            name,
                            ref_audio,
                            ref_text,
                        )
                        return (
                            _voice_library_markup(),
                            gr.update(
                                choices=_saved_voice_choices(),
                                value=saved_name,
                            ),
                            "",
                            None,
                            transcript,
                            f"Voz '{saved_name}' guardada y lista para usar.",
                        )
                    except Exception as e:
                        return (
                            _voice_library_markup(),
                            gr.update(),
                            name or "",
                            ref_audio,
                            ref_text or "",
                            f"Error al guardar: {type(e).__name__}: {e}",
                        )

                library_voice.change(
                    _preview_saved_voice,
                    inputs=[library_voice],
                    outputs=[footer_audio, library_status],
                )
                library_save.click(
                    _save_library_voice_fn,
                    inputs=[library_voice_name, library_ref_audio, library_ref_text],
                    outputs=[
                        library_html,
                        library_voice,
                        library_voice_name,
                        library_ref_audio,
                        library_ref_text,
                        library_status,
                    ],
                )

            def _switch_workspace(
                section,
                current_clone_voice,
                current_audio_voice,
                current_direct_voice,
                current_library_voice,
            ):
                choices = _saved_voice_choices()

                def refresh_choice(current):
                    selected = current if current in choices else no_saved_voice
                    return gr.update(choices=choices, value=selected)

                return (
                    gr.update(visible=section == "Texto a voz"),
                    gr.update(visible=section == "Cambiar voz"),
                    gr.update(visible=section == "Voice Changer"),
                    gr.update(visible=section == "Diseñar voz"),
                    gr.update(visible=section == "Mis voces"),
                    refresh_choice(current_clone_voice),
                    refresh_choice(current_audio_voice),
                    refresh_choice(current_direct_voice),
                    refresh_choice(current_library_voice),
                )

            studio_nav.change(
                _switch_workspace,
                inputs=[
                    studio_nav,
                    vc_saved_voice,
                    st_saved_voice,
                    direct_saved_voice,
                    library_voice,
                ],
                outputs=[
                    clone_view,
                    audio_view,
                    changer_view,
                    design_view,
                    library_view,
                    vc_saved_voice,
                    st_saved_voice,
                    direct_saved_voice,
                    library_voice,
                ],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
