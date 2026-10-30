import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
from download_large import download

ROOT = Path(__file__).resolve().parent
DEST = ROOT / ".cache" / "wheels"
tags = set(sys_tags())
packages = {
    "sympy": "1.14.0", "mpmath": "1.3.0", "networkx": "3.6.1",
    "setuptools": "82.0.1", "accelerate": "1.13.0", "psutil": "7.2.2",
    "hatchling": None, "pathspec": None, "pluggy": None,
    "trove-classifiers": None, "editables": None,
}


def fetch(item):
    name, version = item
    api = f"https://pypi.org/pypi/{name}/" + (f"{version}/" if version else "") + "json"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    for entry in metadata["urls"]:
        if entry["filename"].endswith(".whl") and parse_wheel_filename(entry["filename"])[3] & tags:
            target = DEST / entry["filename"]
            download(entry["url"], target, workers=3)
            with target.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != entry["digests"]["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch: {target}")
            return
    raise RuntimeError(f"No compatible wheel: {name}")


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(fetch, packages.items()))
download(
    "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
    DEST / "torchaudio-2.8.0+cu128-cp312-cp312-win_amd64.whl", workers=3,
)
(ROOT / "logs" / "extras-ready.txt").write_text("ready")
print("EXTRA WHEELS READY", flush=True)
