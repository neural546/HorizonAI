"""Pre-flight health check."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ERRORS: list[str] = []
WARNINGS: list[str] = []


def check(label, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ERRORS.append(f"{label}: {exc}")
        print(f"[FAIL] {label}: {exc}")
        return
    if ok:
        print(f"[ OK ] {label}" + (f" — {detail}" if detail else ""))
    else:
        ERRORS.append(f"{label}: {detail}")
        print(f"[FAIL] {label}: {detail}")


def warn(label, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        WARNINGS.append(f"{label}: {exc}")
        print(f"[WARN] {label}: {exc}")
        return
    if ok:
        print(f"[ OK ] {label}" + (f" — {detail}" if detail else ""))
    else:
        WARNINGS.append(f"{label}: {detail}")
        print(f"[WARN] {label}: {detail}")


def check_imports():
    modules = ["torch", "numpy", "PIL", "yaml", "flask", "imageio",
               "diffusers", "transformers", "accelerate"]
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, f"torch {importlib.import_module('torch').__version__}"


def check_configs():
    expected = ["model.yaml", "training.yaml", "inference.yaml"]
    missing = [f for f in expected if not (ROOT / "config" / f).exists()]
    if missing:
        return False, f"missing configs: {missing}"
    return True, "config/*.yaml present"


def check_tokenizer():
    from AI.tokenizer import build_default_tokenizer
    tok = build_default_tokenizer(max_length=16)
    if tok.vocab_size < 5:
        return False, "tokenizer vocab too small"
    if len(tok.encode("a dragon")) != 16:
        return False, "unexpected encode length"
    return True, f"vocab={tok.vocab_size}"


def check_dataset_module():
    from AI.dataset import validate_dataset
    rep = validate_dataset(ROOT / "tests" / "data")
    return True, f"scanned {rep.total_samples} samples"


def check_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False, "ffmpeg not on PATH"
    return True, exe


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        return False, "CUDA not available (CPU mode will be used)"
    return True, torch.cuda.get_device_name(0)


def check_checkpoints():
    ckpt_dir = ROOT / "AI" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return True, f"{latest.name} present"
    return False, "no latest.pt yet (run training first)"


def check_model_import():
    from AI.model import TextToVideoModel  # noqa: F401
    return True, "import ok"


def check_scheduler():
    from AI.scheduler import DDIMScheduler, DDPMScheduler
    s = DDPMScheduler(num_train_timesteps=100)
    d = DDIMScheduler(num_train_timesteps=100)
    if s.num_train_timesteps != 100 or d.num_train_timesteps != 100:
        return False, "scheduler timestep mismatch"
    return True, "DDPM + DDIM constructed"


def check_frontend():
    for f in ("Frontend/index.html", "static/style.css", "static/app.js"):
        if not (ROOT / f).exists():
            return False, f"missing {f}"
    return True, "frontend files present"


def main() -> int:
    print("=" * 50)
    print("TEXT-TO-VIDEO HEALTH CHECK")
    print("=" * 50)
    check("Python imports", check_imports)
    check("Config files", check_configs)
    check("Tokenizer", check_tokenizer)
    check("Dataset module", check_dataset_module)
    check("Model module", check_model_import)
    check("Schedulers", check_scheduler)
    check("FFmpeg", check_ffmpeg)
    check("Frontend", check_frontend)
    warn("CUDA", check_cuda)
    warn("Checkpoints", check_checkpoints)
    print()
    if ERRORS:
        print("HEALTH CHECK FAILED")
        for e in ERRORS:
            print(f"  - {e}")
        return 1
    print("HEALTH CHECK PASSED")
    if WARNINGS:
        print("Warnings:")
        for w in WARNINGS:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
