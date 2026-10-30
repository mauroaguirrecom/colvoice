"""Verify large downloads against hashes published by their official hosts."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent


def verify(path, expected):
    print(f"Verifying {path.name}...", flush=True)
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")
    print(f"SHA256 OK: {path}", flush=True)


with urllib.request.urlopen("https://download.pytorch.org/whl/cu128/torch/", timeout=60) as response:
    page = response.read().decode()
match = re.search(r'torch-2\.8\.0(?:%2B|\+)cu128-cp312-cp312-win_amd64\.whl#sha256=([0-9a-f]{64})', page)
if not match:
    raise RuntimeError("Official PyTorch hash not found")
verify(ROOT / ".cache" / "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl", match[1])
with urllib.request.urlopen("https://huggingface.co/api/models/k2-fsa/OmniVoice?blobs=true", timeout=60) as response:
    metadata = json.load(response)
for entry in metadata["siblings"]:
    if entry.get("lfs"):
        verify(ROOT / "models" / "OmniVoice" / entry["rfilename"], entry["lfs"]["sha256"])
print("All published hashes match.", flush=True)
