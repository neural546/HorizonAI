#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Text-to-Video AI - Setup"
echo "=========================================="

OS="$(uname -s)"
echo "Detected OS: $OS"

command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }
echo "Python version: $(python3 --version | awk '{print $2}')"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
fi

python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null; then
    echo "CUDA detected - installing CUDA-enabled PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA detected - installing CPU PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

if ! command -v ffmpeg >/dev/null; then
    echo "WARNING: FFmpeg not found (install: sudo apt install ffmpeg OR brew install ffmpeg)"
fi

mkdir -p static/videos outputs AI/checkpoints runs \
    datasets/train datasets/validation tests/data/sample_video

if [ ! -f "tests/data/sample_video/000001.png" ]; then
    echo "Generating sample test frames..."
    python - <<'PY'
import os
from PIL import Image
import numpy as np
out = "tests/data/sample_video"
os.makedirs(out, exist_ok=True)
for i in range(1, 5):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = (i * 30, 100 + i * 20, 200 - i * 20)
    arr[16:48, 16:48] = (255 - i * 30, i * 40, 128)
    Image.fromarray(arr).save(os.path.join(out, f"{i:06d}.png"))
with open(os.path.join(out, "prompt.txt"), "w") as f:
    f.write("a simple colored test scene\n")
print("Sample test frames created.")
PY
fi

echo ""
echo "Running health check..."
python health_check.py || true

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Then run:      python app.py"
echo "=========================================="
