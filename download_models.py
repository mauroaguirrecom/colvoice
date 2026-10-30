"""Download the official public model into this project's models directory."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request
from download_large import download as download_large

ROOT = Path(__file__).resolve().parent / "models" / "OmniVoice"
REPO = "https://huggingface.co/k2-fsa/OmniVoice"


def download(entry):
    name, size = entry["rfilename"], entry.get("size")
    target = ROOT / name
    if target.is_file() and target.stat().st_size == size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if size and size > 100 * 1024 * 1024:
        download_large(f"{REPO}/resolve/main/{name}", target, workers=8)
        return
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {name} ({(size or 0) / 1e6:.1f} MB)", flush=True)
    with urllib.request.urlopen(f"{REPO}/resolve/main/{name}", timeout=120) as source:
        with partial.open("wb") as out:
            while block := source.read(8 * 1024 * 1024):
                out.write(block)
    if size is not None and partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download: {name}")
    partial.replace(target)
    print(f"Ready: {name}", flush=True)


if __name__ == "__main__":
    with urllib.request.urlopen("https://huggingface.co/api/models/k2-fsa/OmniVoice?blobs=true", timeout=60) as response:
        entries = json.load(response)["siblings"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, entries))
    print("Model download complete.", flush=True)
