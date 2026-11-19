"""Small local JSON API for ColVoice agents and automations.

The API intentionally uses only the Python standard library so an agent can
connect without reading or importing the full Gradio interface. It listens on
localhost by default and shares the already-loaded speech model with the UI.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import uuid
from typing import Any

import numpy as np
import soundfile as sf

from omnivoice import OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.utils.audio import load_waveform
from colvoice_quality import analyze_audio_signal, compare_transcripts
from omnivoice.cli.demo import (
    _apply_pronunciation_dictionary,
    _is_spanish_language,
    _prepare_spanish_text,
)


API_VERSION = "1"


class _ColVoiceService:
    def __init__(self, model, root: Path, voice_library: Path):
        self.model = model
        self.root = root.resolve()
        self.voice_library = voice_library.resolve()
        self.output_dir = self.root / "outputs" / "api"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_voices(self) -> list[dict[str, Any]]:
        index_path = self.voice_library / "index.json"
        if not index_path.is_file():
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        voices = []
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            filename = str(entry.get("file", "")).strip()
            if not name or Path(filename).name != filename:
                continue
            if not (self.voice_library / filename).is_file():
                continue
            voices.append(
                {
                    "name": name,
                    "has_reference_audio": bool(
                        entry.get("reference_audio")
                        and (self.voice_library / str(entry["reference_audio"])).is_file()
                    ),
                }
            )
        return sorted(voices, key=lambda item: item["name"].casefold())

    def health(self) -> dict[str, Any]:
        return {
            "brand": "ColVoice",
            "status": "ok",
            "api_version": API_VERSION,
            "model_loaded": self.model is not None,
            "capabilities": ["tts", "voice-cloning", "voice-list", "quality-evaluation"],
        }

    def voices(self) -> dict[str, Any]:
        return {"voices": self._read_voices(), "count": len(self._read_voices())}

    def _find_voice_entry(self, name: str) -> dict[str, Any]:
        wanted = name.strip().casefold()
        for entry in self._read_index_entries():
            if entry["name"].casefold() == wanted:
                result = dict(entry)
                result["prompt"] = VoiceClonePrompt.load(
                    str(self.voice_library / entry["file"])
                )
                return result
        raise ValueError(f"No existe una voz guardada llamada '{name}'.")

    def _read_index_entries(self) -> list[dict[str, str]]:
        index_path = self.voice_library / "index.json"
        if not index_path.is_file():
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [
            entry
            for entry in data
            if isinstance(entry, dict)
            and str(entry.get("name", "")).strip()
            and Path(str(entry.get("file", ""))).name == str(entry.get("file", ""))
            and (self.voice_library / str(entry.get("file", ""))).is_file()
        ] if isinstance(data, list) else []

    def tts(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("El campo 'text' es obligatorio.")
        language = payload.get("language") or "Spanish"
        is_spanish = _is_spanish_language(language)
        target_text = _prepare_spanish_text(text) if is_spanish else text
        voice_entry = None
        voice_name = str(payload.get("voice", "")).strip()
        if voice_name:
            voice_entry = self._find_voice_entry(voice_name)
            if is_spanish:
                target_text = _apply_pronunciation_dictionary(
                    target_text, voice_entry.get("pronunciation", {})
                )
        config = OmniVoiceGenerationConfig(
            num_step=int(payload.get("num_step", 32)),
            guidance_scale=float(payload.get("guidance_scale", 2.0)),
            denoise=bool(payload.get("denoise", True)),
            postprocess_output=True,
        )
        kwargs: dict[str, Any] = {
            "text": target_text,
            "language": language,
            "generation_config": config,
        }
        if is_spanish:
            kwargs["normalize_text"] = True
        if payload.get("speed") is not None:
            kwargs["speed"] = float(payload["speed"])
        if payload.get("duration") is not None:
            kwargs["duration"] = float(payload["duration"])
        if voice_entry:
            kwargs["voice_clone_prompt"] = voice_entry["prompt"]

        audio = self.model.generate(**kwargs)[0]
        waveform = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(waveform).all() or not np.any(np.abs(waveform)):
            raise RuntimeError("El modelo produjo un audio vacío o inválido.")
        output_path = self.output_dir / f"colvoice-api-{uuid.uuid4().hex}.wav"
        sf.write(str(output_path), waveform, self.model.sampling_rate, subtype="PCM_16")
        return {
            "file": str(output_path),
            "relative_file": str(output_path.relative_to(self.root)),
            "sample_rate": self.model.sampling_rate,
            "duration_seconds": round(len(waveform) / self.model.sampling_rate, 3),
            "voice": voice_name or None,
            "language": language,
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Transcribe a local WAV and compare it with the expected script."""
        expected = str(payload.get("expected_text", "")).strip()
        if not expected:
            raise ValueError("El campo 'expected_text' es obligatorio.")
        raw_path = str(payload.get("audio_file", "")).strip()
        if not raw_path:
            raise ValueError("El campo 'audio_file' es obligatorio.")
        audio_path = Path(raw_path).expanduser().resolve()
        if not audio_path.is_file():
            raise ValueError(f"No existe el audio: {audio_path}")
        waveform, sample_rate = load_waveform(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0, keepdims=True)
        if self.model._asr_pipe is None:
            self.model.load_asr_model(device="cpu")
        audio_input = {
            "array": np.squeeze(waveform),
            "sampling_rate": sample_rate,
        }
        try:
            transcript = self.model._asr_pipe(
                audio_input,
                chunk_length_s=30,
                stride_length_s=5,
            )["text"].strip()
        except TypeError:
            transcript = self.model.transcribe((waveform, sample_rate)).strip()
        metrics = compare_transcripts(expected, transcript)
        return {
            "audio_file": str(audio_path),
            "transcript": transcript,
            **analyze_audio_signal(waveform, sample_rate),
            **metrics,
        }


def start_api_server(model, root: Path, voice_library: Path, host="127.0.0.1", port=8002):
    """Start the local API in a daemon thread; return ``None`` if unavailable."""
    service = _ColVoiceService(model, root, voice_library)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, data: dict[str, Any]):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/api/health", "/v1/health"):
                self._send(200, service.health())
            elif self.path in ("/api/voices", "/v1/voices"):
                self._send(200, service.voices())
            else:
                self._send(404, {"error": "Ruta no encontrada."})

        def do_POST(self):  # noqa: N802
            if self.path not in ("/api/tts", "/v1/tts", "/api/evaluate", "/v1/evaluate"):
                self._send(404, {"error": "Ruta no encontrada."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path in ("/api/evaluate", "/v1/evaluate"):
                    self._send(200, service.evaluate(payload))
                else:
                    self._send(200, service.tts(payload))
            except Exception as error:  # API errors are returned as JSON.
                self._send(400, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format, *args):  # noqa: A002
            return

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as error:
        print(f"ColVoice API no disponible en {host}:{port}: {error}", flush=True)
        return None
    thread = threading.Thread(
        target=server.serve_forever,
        name="colvoice-api",
        daemon=True,
    )
    thread.start()
    print(f"ColVoice API: http://{host}:{port}/v1/health", flush=True)
    return server
