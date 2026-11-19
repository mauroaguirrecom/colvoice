"""Local launcher with optional smoke test and conservative memory defaults."""
import argparse
import atexit
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ["PATH"] = str(ROOT / ".venv" / "Scripts") + os.pathsep + os.environ.get("PATH", "")

print("Loading Python libraries...", flush=True)
import torch
from omnivoice import OmniVoice
from omnivoice.cli.demo import build_demo
from colvoice_api import start_api_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--api-port", type=int, default=8002)
    parser.add_argument("--no-api", action="store_true", help="Disable the local agent API.")
    args = parser.parse_args()

    pid_path = ROOT / "logs" / "server.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: pid_path.unlink(missing_ok=True))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoint = str(ROOT / "models" / "OmniVoice")
    asr_model = "openai/whisper-base"
    asr_snapshots = (
        ROOT
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--whisper-base"
        / "snapshots"
    )
    if asr_snapshots.is_dir():
        local_snapshots = sorted(p for p in asr_snapshots.iterdir() if p.is_dir())
        if local_snapshots:
            asr_model = str(local_snapshots[-1])
    print(f"Loading OmniVoice for {args.device}...", flush=True)
    model = OmniVoice.from_pretrained(
        checkpoint, device_map="cpu", dtype=torch.float16,
        load_asr=False, asr_device="cpu", asr_model_name=asr_model,
    )
    if args.device != "cpu":
        # Keep audio encoding/decoding on CPU to leave VRAM for generation.
        audio_tokenizer = model.audio_tokenizer
        model.audio_tokenizer = None
        model.to(args.device)
        model.audio_tokenizer = audio_tokenizer
    print(f"Speech model: {model.device}; audio tokenizer: {model.audio_tokenizer.device}", flush=True)
    print("Model ready.", flush=True)
    if args.test:
        import numpy as np
        import soundfile as sf
        audio = model.generate(
            text="Hola, Mauro. OmniVoice ya funciona en tu equipo.",
            language="es", num_step=16,
        )[0]
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
            raise RuntimeError("The generated audio is invalid or silent.")
        output = ROOT / "outputs" / "prueba.wav"
        output.parent.mkdir(exist_ok=True)
        sf.write(output, audio, model.sampling_rate)
        print(f"Test passed: {output} ({len(audio)/model.sampling_rate:.1f}s)", flush=True)
    demo = build_demo(model, checkpoint, voice_library_dir=ROOT / "voices")
    api_server = None
    if not args.no_api:
        api_server = start_api_server(
            model,
            ROOT,
            ROOT / "voices",
            host="127.0.0.1",
            port=args.api_port,
        )
    try:
        demo.queue(default_concurrency_limit=1).launch(
            server_name="127.0.0.1", server_port=args.port, share=False,
        )
    finally:
        if api_server is not None:
            api_server.shutdown()


if __name__ == "__main__":
    main()
