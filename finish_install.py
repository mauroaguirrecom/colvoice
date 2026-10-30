"""Finish this isolated installation after the large downloads complete."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
WHEEL = ROOT / ".cache" / "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"
deadline = time.monotonic() + 3600
print("Waiting for the PyTorch wheel and initial dependencies...", flush=True)
while True:
    log = ROOT / "logs" / "dependencies.log"
    installed = log.exists() and "Successfully installed" in log.read_text(errors="replace")
    if WHEEL.exists() and installed and (ROOT / "logs" / "extras-ready.txt").exists():
        break
    if time.monotonic() > deadline:
        raise RuntimeError("Downloads or dependency installation did not finish")
    time.sleep(5)

with urllib.request.urlopen("https://download.pytorch.org/whl/cu128/torch/", timeout=60) as response:
    page = response.read().decode()
match = re.search(r'torch-2\.8\.0(?:%2B|\+)cu128-cp312-cp312-win_amd64\.whl#sha256=([0-9a-f]{64})', page)
if not match:
    raise RuntimeError("Official PyTorch SHA256 not found")
with WHEEL.open("rb") as source:
    actual = hashlib.file_digest(source, "sha256").hexdigest()
if actual != match[1]:
    raise RuntimeError("Downloaded PyTorch SHA256 does not match")
print("PyTorch SHA256 verified.", flush=True)


def pip(*arguments):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(ROOT / ".cache" / "wheels"), *arguments], cwd=ROOT, check=True)


pip(str(WHEEL))
pip("torchaudio==2.8.0+cu128", "accelerate==1.13.0", "hatchling", "editables")
pip("--no-build-isolation", "-e", "./app")
import imageio_ffmpeg
shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ROOT / ".venv" / "Scripts" / "ffmpeg.exe")
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
with (ROOT / "installed-versions.txt").open("w") as output:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=output, check=True)
(ROOT / "logs" / "install-complete.txt").write_text("OmniVoice installed successfully.\n")
print("INSTALL COMPLETE", flush=True)
