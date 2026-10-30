"""Resumable HTTP range downloader for large public model/runtime files."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
import urllib.request


def download(url, destination, workers=6):
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    if target.exists() and target.stat().st_size == size:
        print(f"Already complete: {target}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    state_path = target.with_name(target.name + ".progress.json")
    chunk = 8 * 1024 * 1024
    state = json.loads(state_path.read_text()) if state_path.exists() else {"size": size, "done": []}
    if state["size"] != size:
        raise RuntimeError("Remote size changed; use a fresh destination")
    completed = set(state["done"])
    if not partial.exists():
        completed.clear()
    with partial.open("ab") as out:
        out.truncate(size)
    lock = threading.Lock()
    count = (size + chunk - 1) // chunk
    start = time.monotonic()

    def fetch(index):
        if index in completed:
            return
        lo, hi = index * chunk, min((index + 1) * chunk, size) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise RuntimeError(f"Server did not honor range: {response.status}")
                    data = response.read()
                if len(data) != hi - lo + 1:
                    raise RuntimeError("Incomplete chunk")
                with lock:
                    with partial.open("r+b") as out:
                        out.seek(lo)
                        out.write(data)
                    completed.add(index)
                    state_path.write_text(json.dumps({"size": size, "done": sorted(completed)}))
                    print(f"{target.name}: {len(completed)}/{count} blocks ({100 * len(completed)/count:.1f}%), {time.monotonic()-start:.0f}s", flush=True)
                return
            except Exception as exc:
                print(f"Retry {attempt + 1}, block {index}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range(count)))
    partial.replace(target)
    state_path.unlink(missing_ok=True)
    print(f"Complete: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    download(args.url, args.destination, args.workers)
