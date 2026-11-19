"""Cross-platform ColVoice bootstrap for people and coding agents.

Usage:
    python setup_colvoice.py              # auto-detect NVIDIA, otherwise CPU/MPS
    python setup_colvoice.py --cuda      # force the CUDA 12.8 wheels
    python setup_colvoice.py --cpu       # force the portable CPU/MPS wheels
    python setup_colvoice.py --skip-model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def venv_python(venv: Path) -> Path:
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not candidate.is_file():
        raise RuntimeError(f"No se encontró el intérprete del entorno: {candidate}")
    return candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Instalador multiplataforma de ColVoice")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cuda", action="store_true", help="Instalar PyTorch CUDA 12.8.")
    backend.add_argument("--cpu", action="store_true", help="Instalar PyTorch portable CPU/MPS.")
    parser.add_argument("--skip-model", action="store_true", help="No descargar los pesos todavía.")
    args = parser.parse_args(argv)

    if sys.version_info < (3, 10):
        raise SystemExit("ColVoice requiere Python 3.10 o superior.")

    use_cuda = args.cuda or (not args.cpu and shutil.which("nvidia-smi") is not None)
    venv = ROOT / ".venv"
    if not venv.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    python = venv_python(venv)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])

    if use_cuda:
        run([
            str(python), "-m", "pip", "install",
            "-r", str(ROOT / "requirements-cuda.txt"),
        ])
    else:
        run([
            str(python), "-m", "pip", "install",
            "torch>=2.8.0", "torchaudio>=2.8.0",
        ])
        run([
            str(python), "-m", "pip", "install",
            "-r", str(ROOT / "requirements-local.txt"),
        ])

    if not args.skip_model:
        run([str(python), str(ROOT / "download_models.py")])
    print("ColVoice está listo. Ejecuta la aplicación con el launcher de tu sistema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
