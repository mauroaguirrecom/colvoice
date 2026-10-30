"""Fetch pip's already-resolved wheels using resumable range downloads."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
from download_large import download

ROOT = Path(__file__).resolve().parent
DEST = ROOT / ".cache" / "wheels"
DEST.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs" / "dependencies.log"
names = sorted(set(re.findall(r"Downloading (\S+\.whl)\.metadata", LOG.read_text())))


def fetch(filename):
    name, version = filename.split("-")[:2]
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
        info = json.load(response)
    entry = next(item for item in info["urls"] if item["filename"] == filename)
    target = DEST / filename
    download(entry["url"], target, workers=3)
    with target.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != entry["digests"]["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {filename}")
    return target


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    wheels = list(pool.map(fetch, names))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", *map(str, wheels)], check=True)
with LOG.open("a") as log:
    log.write("\nSuccessfully installed all resolved dependencies via verified local wheels.\n")
print("DEPENDENCIES READY", flush=True)
