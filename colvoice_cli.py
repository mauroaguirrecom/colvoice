"""Token-efficient command line client for the ColVoice local API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(api_url: str, path: str, payload=None):
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(api_url.rstrip("/") + path, data=body, headers=headers)
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API {error.code}: {detail}") from error
    except URLError as error:
        raise SystemExit(
            "No se pudo conectar con ColVoice. Inicia la app con "
            "Iniciar-ColVoice.cmd y verifica el puerto 8002."
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cliente CLI compacto para ColVoice; no necesita leer demo.py."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8002/v1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Comprobar que ColVoice está listo.")
    voices = subparsers.add_parser("voices", help="Consultar voces guardadas.")
    voices.add_argument("action", choices=["list"], default="list", nargs="?")

    tts = subparsers.add_parser("tts", help="Generar un WAV desde texto.")
    tts.add_argument("--text", required=True)
    tts.add_argument("--voice", help="Nombre exacto de una voz guardada.")
    tts.add_argument("--language", default="Spanish")
    tts.add_argument("--output", type=Path, required=True)
    tts.add_argument("--speed", type=float)
    tts.add_argument("--duration", type=float)
    tts.add_argument("--num-step", type=int, default=32)
    tts.add_argument("--guidance-scale", type=float, default=2.0)

    evaluate = subparsers.add_parser(
        "evaluate", help="Comparar un audio con el texto que debía pronunciar."
    )
    evaluate.add_argument("--audio", type=Path, required=True)
    evaluate.add_argument("--text", required=True, help="Texto esperado por el audio.")

    args = parser.parse_args(argv)
    if args.command == "health":
        print(json.dumps(_request(args.api_url, "/health"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "voices":
        print(json.dumps(_request(args.api_url, "/voices"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "tts":
        payload = {
            "text": args.text,
            "voice": args.voice,
            "language": args.language,
            "speed": args.speed,
            "duration": args.duration,
            "num_step": args.num_step,
            "guidance_scale": args.guidance_scale,
        }
        result = _request(args.api_url, "/tts", payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result["file"], args.output)
        print(json.dumps({**result, "output": str(args.output)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        result = _request(
            args.api_url,
            "/evaluate",
            {"audio_file": str(args.audio.resolve()), "expected_text": args.text},
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
