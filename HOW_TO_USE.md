# How To Use — Text-to-Video AI

A practical, copy-paste guide for running every part of the system.

> **Companion docs:**
> - `README.md` — project overview, architecture, troubleshooting table
> - `HOW_IT_WORKS.md` — internal design, every function explained

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verify the Install](#3-verify-the-install)
4. [Prepare a Dataset](#4-prepare-a-dataset)
5. [Validate and Repair the Dataset](#5-validate-and-repair-the-dataset)
6. [Train the Full Model](#6-train-the-full-model)
7. [Resume Training](#7-resume-training)
8. [Run Inference (CLI)](#8-run-inference-cli)
9. [Use the Lightweight MVP](#9-use-the-lightweight-mvp)
10. [Use the Web UI](#10-use-the-web-ui)
11. [Use the REST API](#11-use-the-rest-api)
12. [Use Docker](#12-use-docker)
13. [Choose a Resource Profile](#13-choose-a-resource-profile)
14. [Run Tests](#14-run-tests)
15. [Recipes](#15-recipes)
16. [Troubleshooting Checklist](#16-troubleshooting-checklist)
17. [Cheat Sheet](#17-cheat-sheet)

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.11 | `python3 --version` |
| pip | latest | `python3 -m pip install --upgrade pip` |
| FFmpeg | any recent | `ffmpeg -version` — required for MP4 export |
| Disk | ~5 GB | PyTorch wheels + checkpoints |
| RAM | 8 GB | 16 GB+ recommended |
| GPU (optional) | 4 GB VRAM | CPU works via the MVP path |
| OS | Linux / macOS / Windows | Linux + macOS tested; Windows via Git Bash |

Install FFmpeg:

    sudo apt update && sudo apt install -y ffmpeg     # Ubuntu/Debian
    brew install ffmpeg                               # macOS
    choco install ffmpeg                              # Windows

CUDA (only if you have an NVIDIA GPU): install the CUDA toolkit that matches your driver. `setup.sh` picks the right PyTorch wheel automatically.

---

## 2. Installation

Pick one of three paths.

### Path A — Script (recommended)

    chmod +x setup.sh
    ./setup.sh

The script:

1. Creates `.venv/`
2. Detects CUDA (via `nvidia-smi`) and installs the matching PyTorch wheel
3. Installs everything in `requirements.txt`
4. Creates `static/videos/`, `outputs/`, `AI/checkpoints/`, `runs/`, `datasets/{train,validation}/`
5. Generates 4 sample frames in `tests/data/sample_video/`
6. Runs `health_check.py`

### Path B — Manual

    python3 -m venv .venv
    source .venv/bin/activate         # Linux/macOS
    # source .venv/Scripts/activate   # Windows Git Bash

    python -m pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    # or, for CUDA 12.1:
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    pip install -r requirements.txt

### Path C — Docker

    docker compose up --build

Open http://localhost:5000. See §12.

---

## 3. Verify the Install

    source .venv/bin/activate
    python health_check.py

Expected tail:

    HEALTH CHECK PASSED

If it fails, the output lists exactly which check failed.

Common failures:

| Failure | Fix |
|---------|-----|
| `missing: torch` | `pip install -r requirements.txt` |
| `ffmpeg not on PATH` | install FFmpeg (§1) |
| `missing configs` | restore `config/` |
| `missing Frontend/index.html` or `static/*` | restore those files |

Warnings (not failures): missing CUDA, missing `latest.pt`. Both are expected before your first training run.

Quick end-to-end sanity check (no training needed):

    pytest tests/smoke -v

All four smoke tests should pass in under 60 s on CPU.

---

## 4. Prepare a Dataset

The model reads one directory per sample:

    datasets/train/sample0001/
        prompt.txt
        000001.png
        000002.png
        000003.png
        ...

Rules:

| Rule | Detail |
|------|--------|
| Prompt file | Exactly one `prompt.txt`, one line, non-empty |
| Frame files | Named `NNNNNN.{png,jpg,jpeg}` starting at `000001` |
| Numbering | Strictly sequential, zero gaps, zero duplicates |
| Count | 8 ≤ frames ≤ 1024 |
| Resolution | Every frame in a sample must be the same size |
| Channels | RGB; grayscale/RGBA are auto-converted during validation |

Example (3-sample dataset):

    mkdir -p datasets/train/sample0001 datasets/train/sample0002 datasets/train/sample0003

    echo "a dragon flying above snowy mountains"  > datasets/train/sample0001/prompt.txt
    echo "a cat walking through a forest"         > datasets/train/sample0002/prompt.txt
    echo "a spaceship travelling through stars"   > datasets/train/sample0003/prompt.txt

    # Copy your frames into each directory as 000001.png, 000002.png, ...

Validation data uses the same structure under `datasets/validation/`. The training script only reads `datasets/train/` unless you pass `--data-dir`.

---

## 5. Validate and Repair the Dataset

### Validate

    python AI/dataset.py --validate datasets/train

Success (exit code 0):

    Dataset Summary
    ===============
    Root:            datasets/train
    Total Samples:   3
    Valid Samples:   3
    Invalid Samples: 0

Failure (exit code 1):

    Dataset Summary
    ===============
    ...
    Invalid Samples: 1

    Error Report
    ------------
      sample0002:
        - too few frames: 5 < 8
        - non-sequential numbering: expected 000006, found 000007

Common error messages:

| Error | Meaning |
|-------|---------|
| `missing prompt.txt` | Sample dir has no prompt |
| `prompt.txt is empty` | Prompt file contains only whitespace |
| `no image frames found` | No files matching `NNNNNN.{png,jpg,jpeg}` |
| `too few frames: N < 8` | Add frames or drop the sample |
| `non-sequential numbering` | A number is missing; run `--repair` |
| `unsupported file: X` | Non-image file in the sample dir |
| `unreadable/corrupt image: X` | PIL can't decode it; run `--repair` |
| `resolution mismatch in X` | Frames have different sizes |
| `duplicate frame detected: X` | Two frames are byte-identical |

### Repair

    python AI/dataset.py --repair datasets/train

What it does:

- Deletes unsupported and corrupt files
- Renumbers remaining frames to be sequential (`000001`, `000002`, …)
- Removes samples that still have no prompt or no frames

Example output:

    Repair Report
    =============
    removed_files: 1
      - datasets/train/sample0002/000004.png
    renamed: 5
      - datasets/train/sample0002/000003.png
      ...
    removed_samples: 0

Re-run validation afterwards:

    python AI/dataset.py --validate datasets/train

Typical loop:

    python AI/dataset.py --validate datasets/train   # see what's wrong
    python AI/dataset.py --repair   datasets/train   # fix what can be auto-fixed
    python AI/dataset.py --validate datasets/train   # confirm

Note: repair cannot fix resolution mismatch or too few frames. Add/resize frames manually.

---

## 6. Train the Full Model

    python AI/train.py --profile low --seed 42

What happens:

1. Merges `config/model.yaml` + `config/training.yaml` for the `low` profile
2. Seeds all RNGs
3. Loads or creates `AI/checkpoints/tokenizer.json`
4. Iterates `datasets/train/` in batches
5. For each batch: encodes frames to latents → adds noise → asks the model to predict it → MSE loss → AdamW step
6. Writes `AI/checkpoints/latest.pt` every `save_every` steps

Expected console:

    Profile: low | device=cpu
    Dataset: datasets/train
    Vocab size: 87
    Samples: 3
    Starting training...
    Epoch 1/10: 100%|█████████████████| 3/3 [00:42<00:00, loss=0.4123, step=3]
    Epoch 1 average loss: 0.4123
    ...
    Training complete.

Config you'll likely want to change:

| Want | Change |
|------|--------|
| More epochs | `--epochs 50` or edit `config/training.yaml` `defaults.epochs` |
| Different LR | edit `config/training.yaml` `defaults.learning_rate` |
| Different batch size | use a bigger profile, or edit that profile's `batch_size` |
| Save more often | `config/training.yaml` `defaults.save_every` |

Monitor with TensorBoard (optional):

    tensorboard --logdir runs --host 0.0.0.0 --port 6006

Open http://localhost:6006.

Checkpoints land in:

    AI/checkpoints/
    ├── tokenizer.json
    ├── latest.pt
    └── step_000500.pt

---

## 7. Resume Training

    python AI/train.py --resume AI/checkpoints/latest.pt

The script reads `epoch` and `global_step` from the checkpoint and continues from there. Optimizer state is also restored.

Combine with other flags:

    python AI/train.py --resume AI/checkpoints/latest.pt \
                       --epochs 20 \
                       --profile low \
                       --seed 42

To start fresh but keep the vocab, delete only the `.pt` files:

    rm AI/checkpoints/latest.pt AI/checkpoints/step_*.pt
    python AI/train.py --profile low --seed 42

To rebuild the vocab from a new dataset, delete `tokenizer.json` too. Training will rebuild and save a new one — but any previously saved model will be incompatible.

---

## 8. Run Inference (CLI)

Once you have a checkpoint:

    python AI/inference.py \
        --prompt "a dragon flying above snowy mountains" \
        --seed 42

Output: `static/videos/output.mp4` (H.264, 8 fps by default).

Full flag reference:

| Flag | Default | Notes |
|------|---------|-------|
| `--prompt` | required | the text description |
| `--seed` | 42 | reproducibility |
| `--profile` | auto-detect | `low` / `medium` / `high` |
| `--auto-profile` | off | detect hardware |
| `--cfg-scale` | 7.5 | CFG (1.0–20.0) |
| `--num-frames` | profile | frames to generate |
| `--num-inference-steps` | profile | DDIM steps |
| `--image-size` | profile | 64 / 128 / 256 |
| `--fps` | profile | output frame rate |
| `--output` | `static/videos/output.mp4` | output path |
| `--checkpoint` | `AI/checkpoints/latest.pt` | alternate weights |
| `--device` | auto | `cpu` or `cuda` |

Examples:

    # Fast / small
    python AI/inference.py --prompt "a cat walking" \
        --image-size 64 --num-frames 8 --num-inference-steps 20

    # Higher quality
    python AI/inference.py --prompt "a dragon at sunset" \
        --image-size 128 --num-frames 16 --num-inference-steps 50 --cfg-scale 9

    # Reproducible rerun
    python AI/inference.py --prompt "a cat" --seed 123 --output static/videos/cat.mp4
    python AI/inference.py --prompt "a cat" --seed 123 --output static/videos/cat2.mp4
    # cat.mp4 and cat2.mp4 will be identical

    # Force CPU
    python AI/inference.py --prompt "a cat" --device cpu

Wrappers (`train.sh` / `generate.sh`):

    ./generate.sh "a dragon flying over mountains"
    ./generate.sh "a dragon flying over mountains" --cfg 9 --frames 16 --steps 40

---

## 9. Use the Lightweight MVP

The MVP is the fast path. No dataset required. Fits on CPU or a 4 GB GPU.

Train (synthetic data, ~30 s on CPU):

    python AI/lightweight/train.py --steps 5 --seed 42

Output:

    [MVP] Device: cpu
    [MVP] step 1/5 loss=0.9812
    ...
    [MVP] trained 5 steps in 12.34s
    [MVP] saved checkpoint to AI/checkpoints/lightweight_latest.pt

Writes `tokenizer.json` and `lightweight_latest.pt`.

Run MVP inference:

    python AI/lightweight/inference.py \
        --prompt "a simple test scene" \
        --seed 42

Output: `static/videos/mvp_output.mp4`.

Flags: `--cfg-scale`, `--num-inference-steps`, `--num-frames`, `--image-size`, `--fps`, `--seed`, `--output`.

Target: under 30 s on CPU for 64×64, 8 frames, 20 steps.

---

## 10. Use the Web UI

    python app.py

Open http://localhost:5000.

Workflow:

1. Type a prompt
2. Adjust CFG Scale / Frames / Resolution / FPS / Seed
3. Click Generate Video
4. Watch the progress bar advance through the DDIM loop
5. The generated MP4 auto-plays in the Result card
6. Click Download MP4

Behind the scenes:

    POST /generate  → 202 {job_id}
    Browser polls GET /progress?job_id=… every 1.2 s
    Worker thread: load model → run DDIM + CFG → export MP4
    /progress returns {status:"completed", video_url:"/video/<job>.mp4"}
    <video src="/video/<job>.mp4">

Shortcut: `GET /video` always serves the newest `.mp4` in `static/videos/`.

---

## 11. Use the REST API

Generate:

    curl -X POST http://localhost:5000/generate \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "a dragon flying in the sky",
        "cfg_scale": 7.5,
        "num_frames": 8,
        "image_size": 64,
        "fps": 8,
        "seed": 42
      }'

Response (202):

    { "job_id": "abc123def456", "status": "queued" }

Poll:

    curl "http://localhost:5000/progress?job_id=abc123def456"

Response while running:

    {
      "job_id": "abc123def456",
      "status": "running",
      "progress": 0.42,
      "message": "sampling 8/20",
      "video_url": null,
      "error": null
    }

Response when done:

    {
      "job_id": "abc123def456",
      "status": "completed",
      "progress": 1.0,
      "message": "done",
      "video_url": "/video/abc123def456.mp4",
      "error": null
    }

Download:

    curl -o out.mp4 http://localhost:5000/video/abc123def456.mp4

Validation rules for /generate (return 400 on failure):

| Field | Rule |
|-------|------|
| `prompt` | non-empty string |
| `cfg_scale` | 1.0 ≤ x ≤ 20.0 |
| `num_frames` | 8 ≤ x ≤ 1024 |
| `image_size` | 64, 128, or 256 |
| `fps` | 1 ≤ x ≤ 60 |

Unparseable numbers also yield 400.

End-to-end script:

    #!/usr/bin/env bash
    set -euo pipefail

    JOB=$(curl -s -X POST http://localhost:5000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt":"a cat walking","seed":42}' \
        | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

    echo "Job: $JOB"

    while true; do
        STATUS=$(curl -s "http://localhost:5000/progress?job_id=$JOB" \
            | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
        echo "  $STATUS"
        [ "$STATUS" = "completed" ] && break
        [ "$STATUS" = "failed" ]    && { echo "failed"; exit 1; }
        sleep 1
    done

    curl -o video.mp4 "http://localhost:5000/video/$JOB.mp4"
    echo "Saved video.mp4"

---

## 12. Use Docker

Start:

    docker compose up --build

Open http://localhost:5000.

The container installs FFmpeg + Python 3.11 + PyTorch (CPU) + requirements,
mounts `./datasets`, `./outputs`, `./AI/checkpoints`, `./static/videos`,
exposes port 5000, and polls `/health` every 30 s.

Train / infer inside Docker:

    docker compose exec text-to-video python AI/train.py --profile low --seed 42
    docker compose exec text-to-video python AI/inference.py --prompt "a dragon"

With TensorBoard:

    docker compose --profile monitoring up

TensorBoard: http://localhost:6006.

Stop:

    docker compose down

GPU-enabled Docker (optional) — edit `Dockerfile` to install the CUDA wheel and add to `docker-compose.yml`:

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

Then rebuild.

---

## 13. Choose a Resource Profile

| Profile | VRAM | Image | Frames | Batch | Best for |
|---------|------|-------|--------|-------|----------|
| low | 4 GB | 64×64 | 8 | 1 | CPU fallback / small GPU |
| medium | 8 GB | 128×128 | 16 | 2 | mid-range GPU |
| high | 16 GB+ | 256×256 | 32 | 8 | workstation |

Explicit profile:

    python AI/train.py   --profile medium --seed 42
    python AI/inference.py --prompt "..." --profile medium

Auto-detect:

    python AI/train.py   --auto-profile --seed 42
    python AI/inference.py --prompt "..." --auto-profile

Manual inspection:

    python AI/resource_detector.py

    Resource Report
    ===============
    CPU count:     8
    RAM (GB):      16.0
    CUDA available:False
    VRAM (GB):     0.0
    Suggested profile: low

Override inside a profile:

    python AI/inference.py --prompt "a cat" --profile low --num-frames 16

The explicit flag wins over the profile's value.

---

## 14. Run Tests

Unit tests (~2–3 min on CPU):

    pytest -v

Covers tokenizer, dataset, collate, scheduler (DDPM + DDIM), model shapes, inference seeding, export (FFmpeg), Flask API contract, and regressions (num_res_blocks ≥ 2, DDIM 0-d expand, single-step timestep list, gradient checkpointing).

Smoke tests (~30–60 s on CPU):

    pytest tests/smoke -v

End-to-end fast path: MVP training (2 steps), MVP inference + MP4 export (3 steps), raw MP4 export, Flask startup + /generate + /progress.

Single test:

    pytest tests/test_scheduler.py::test_ddim_deterministic -v

Skip FFmpeg tests:

    pytest -v -k "not ffmpeg"

Lint locally:

    ruff check .
    black --check .
    isort --check-only .

---

## 15. Recipes

Generate right now without training:

    python AI/lightweight/train.py --steps 3 --seed 42
    python AI/lightweight/inference.py --prompt "a simple test scene" --seed 42
    open static/videos/mvp_output.mp4     # macOS; xdg-open on Linux

Specific look:

    python AI/inference.py \
        --prompt "a dragon flying above snowy mountains at sunset, cinematic" \
        --cfg-scale 12 \
        --num-inference-steps 40 \
        --image-size 128 \
        --num-frames 16 \
        --seed 777

Reproducible rerun:

    python AI/inference.py --prompt "a cat" --seed 2024 --output static/videos/a.mp4
    python AI/inference.py --prompt "a cat" --seed 2024 --output static/videos/b.mp4
    cmp static/videos/a.mp4 static/videos/b.mp4 && echo "identical"

From raw image folder to video:

    mkdir -p datasets/train/sample0001
    i=1
    for f in my_frames/*.png; do
        cp "$f" "$(printf 'datasets/train/sample0001/%06d.png' $i)"
        i=$((i+1))
    done
    echo "a cinematic scene" > datasets/train/sample0001/prompt.txt
    python AI/dataset.py --validate datasets/train
    python AI/train.py --profile low --epochs 1 --seed 42
    python AI/inference.py --prompt "a cinematic scene" --seed 42

Different port:

    PORT=8080 python app.py

Alternate checkpoint:

    python AI/inference.py --checkpoint AI/checkpoints/step_000500.pt --prompt "a cat"

Batch-generate:

    for p in "a dragon" "a cat walking" "a spaceship"; do
        slug=$(echo "$p" | tr ' ' '_')
        python AI/inference.py --prompt "$p" --seed 42 \
            --output "static/videos/${slug}.mp4"
    done

Reduce VRAM:

    python AI/train.py --profile low --seed 42
    # low profile already sets use_gradient_checkpointing: true

Speed up inference:

    python AI/inference.py --prompt "..." --num-inference-steps 10
    # or use the MVP entirely

---

## 16. Troubleshooting Checklist

Go top-to-bottom; the first failing check is usually the cause.

    [ ] 1. Is Python 3.11+ active?     python --version
    [ ] 2. Is the venv activated?      which python  → should point into .venv/
    [ ] 3. Is FFmpeg installed?        ffmpeg -version
    [ ] 4. Does health check pass?     python health_check.py
    [ ] 5. Does the dataset validate?  python AI/dataset.py --validate datasets/train
    [ ] 6. Is a checkpoint present?    ls AI/checkpoints/latest.pt
    [ ] 7. Does inference run?         python AI/inference.py --prompt "a cat"
    [ ] 8. Does Flask start?           python app.py  →  http://localhost:5000
    [ ] 9. Do smoke tests pass?        pytest tests/smoke -v
    [ ] 10. Do unit tests pass?        pytest -v

Symptom → Cause → Fix:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: torch` | venv not active | `source .venv/bin/activate` |
| `FFmpeg not found on PATH` | FFmpeg not installed | §1 |
| `HEALTH CHECK FAILED — missing: diffusers` | partial install | `pip install -r requirements.txt` |
| `Dataset validation failed` | bad layout | `python AI/dataset.py --repair datasets/train` |
| `ERROR: no valid samples in dataset` | all samples invalid | `python AI/dataset.py --validate datasets/train` |
| Inference runs, output is noise | random weights (no `latest.pt`) | train first, or run MVP training |
| Inference runs, output is static | too few inference steps | `--num-inference-steps 40` |
| `CUDA out of memory` | model too big | `--profile low`, lower `--image-size` / `--num-frames` |
| Flask returns 400 | invalid JSON | check prompt, cfg_scale, num_frames, image_size, fps |
| Flask returns 404 on /video/<id> | wrong job id or file overwritten | `GET /video` returns newest |
| Job stuck at running | worker crashed | check terminal for traceback; job flips to failed |
| Web UI has no styling | CSS not served | confirm `static/style.css`; `curl http://localhost:5000/static/style.css` |
| `Address already in use` | port 5000 taken | `PORT=8080 python app.py` |
| Docker build fails at libx264 | FFmpeg not in image | rebuild; Dockerfile installs FFmpeg |
| `pytest` extremely slow | MVP path being hit repeatedly | `pytest -v -k "not smoke"` for unit suite only |

---

## 17. Cheat Sheet

Setup:

    ./setup.sh
    python health_check.py

Dataset:

    python AI/dataset.py --validate datasets/train
    python AI/dataset.py --repair   datasets/train

Training:

    python AI/train.py --profile low --seed 42
    python AI/train.py --auto-profile --seed 42
    python AI/train.py --resume AI/checkpoints/latest.pt
    python AI/train.py --epochs 20 --profile medium --seed 42

Inference:

    python AI/inference.py --prompt "a dragon flying" --seed 42
    python AI/inference.py --prompt "..." --cfg-scale 9 --num-frames 16 \
                            --image-size 128 --num-inference-steps 40
    python AI/inference.py --prompt "..." --checkpoint AI/checkpoints/step_000500.pt

MVP:

    python AI/lightweight/train.py --steps 5 --seed 42
    python AI/lightweight/inference.py --prompt "a simple test scene" --seed 42

Flask:

    python app.py
    PORT=8080 python app.py

Docker:

    docker compose up --build
    docker compose exec text-to-video python AI/inference.py --prompt "..."
    docker compose --profile monitoring up
    docker compose down

Tests:

    pytest -v
    pytest tests/smoke -v
    pytest tests/test_scheduler.py -v
    pytest -v -k "not ffmpeg"

Resource inspection:

    python AI/resource_detector.py

Cleanup:

    rm AI/checkpoints/latest.pt AI/checkpoints/step_*.pt
    rm AI/checkpoints/tokenizer.json
    rm static/videos/*.mp4
    rm -rf runs/

---

## Recommended First-Run Order

    # 1. Install
    ./setup.sh

    # 2. Verify
    python health_check.py

    # 3. Fast demo (no training needed)
    pytest tests/smoke -v
    python AI/lightweight/train.py --steps 3 --seed 42
    python AI/lightweight/inference.py --prompt "a simple test scene" --seed 42

    # 4. Web UI
    python app.py   # open http://localhost:5000

    # 5. Real training (needs datasets/train/)
    python AI/dataset.py --validate datasets/train
    python AI/train.py --profile low --seed 42
    python AI/inference.py --prompt "a dragon flying" --seed 42

If all ten troubleshooting checks in §16 pass, the repository is fully functional end-to-end.
