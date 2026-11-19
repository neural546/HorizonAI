"""Flask backend for the text-to-video AI system."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

app = Flask(
    __name__,
    static_folder=str(_ROOT / "static"),
    static_url_path="/static",
    template_folder=str(_ROOT / "Frontend"),
)

JOBS: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
JOB_TTL_SECONDS = 3600

MODEL_CACHE: "OrderedDict[Any, Any]" = OrderedDict()
MODEL_CACHE_LOCK = threading.Lock()
MAX_MODEL_CACHE = 3


def _prune_jobs() -> None:
    now = time.time()
    stale = [jid for jid, j in JOBS.items() if now - j.get("created_at", now) > JOB_TTL_SECONDS]
    for jid in stale:
        JOBS.pop(jid, None)


def _new_job(prompt, cfg_scale, num_frames, image_size, fps, seed) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOB_LOCK:
        _prune_jobs()
        JOBS[job_id] = {
            "id": job_id, "status": "queued", "progress": 0.0, "message": "queued",
            "prompt": prompt, "cfg_scale": cfg_scale, "num_frames": num_frames,
            "image_size": image_size, "fps": fps, "seed": seed,
            "video_url": None, "error": None, "created_at": time.time(),
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _tokenizer_fingerprint(tokenizer) -> str:
    items = sorted(tokenizer.token_to_id.items(), key=lambda kv: kv[1])
    payload = "|".join(f"{k}:{v}" for k, v in items).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _get_or_build_model(device, image_size, num_frames, tokenizer, m_cfg, model_cfg):
    key = (str(device), int(image_size), int(num_frames),
           tokenizer.vocab_size, _tokenizer_fingerprint(tokenizer))
    with MODEL_CACHE_LOCK:
        if key in MODEL_CACHE:
            MODEL_CACHE.move_to_end(key)
            return MODEL_CACHE[key]
        from AI.model import TextToVideoModel
        import torch
        model = TextToVideoModel(
            vocab_size=tokenizer.vocab_size, image_size=image_size, num_frames=num_frames,
            latent_channels=m_cfg["latent_channels"], base_channels=m_cfg["base_channels"],
            channel_multipliers=tuple(m_cfg["channel_multipliers"]),
            num_res_blocks=m_cfg["num_res_blocks"], text_embed_dim=m_cfg["text_embed_dim"],
            text_max_length=m_cfg["text_max_length"], transformer_layers=m_cfg["transformer_layers"],
            transformer_heads=m_cfg["transformer_heads"], attention_heads=m_cfg["attention_heads"],
            dropout=m_cfg["dropout"], vae_downsample=model_cfg["vae"]["downsample_factor"],
        ).to(device)
        ckpt = _ROOT / "AI" / "checkpoints" / "latest.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
        model.eval()
        MODEL_CACHE[key] = model
        while len(MODEL_CACHE) > MAX_MODEL_CACHE:
            MODEL_CACHE.popitem(last=False)
        return model


def _run_generation(job_id: str) -> None:
    job = _get_job(job_id)
    if job is None:
        return
    # TESTING short-circuit: skip real inference on CI runners
    if app.config.get("TESTING"):
        _update_job(job_id, status="completed", progress=1.0,
                    message="test-mode completed",
                    video_url="/video/test.mp4")
        return
    try:
        _update_job(job_id, status="running", progress=0.02, message="loading model")

        from AI.export import export_video
        from AI.inference import (generate_video, load_tokenizer, load_yaml,
                                  resolve_profile, set_global_seed)
        from AI.scheduler import DDIMScheduler
        import torch

        seed = int(job["seed"])
        set_global_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = load_yaml(_ROOT / "config" / "model.yaml")
        inference_cfg = load_yaml(_ROOT / "config" / "inference.yaml")
        m_cfg, i_cfg, _ = resolve_profile(model_cfg, inference_cfg, None, auto_profile=True)

        image_size = int(job["image_size"])
        num_frames = int(job["num_frames"])
        tokenizer = load_tokenizer(_ROOT / "AI" / "checkpoints", m_cfg["text_max_length"])
        model = _get_or_build_model(device, image_size, num_frames, tokenizer, m_cfg, model_cfg)

        _update_job(job_id, progress=0.05, message="model loaded")

        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                  device=device, eta=0.0)

        def _cb(step: int, total: int) -> None:
            pct = 0.05 + 0.85 * (step / max(1, total))
            _update_job(job_id, progress=pct, message=f"sampling {step}/{total}")

        frames = generate_video(
            prompt=str(job["prompt"]), model=model, tokenizer=tokenizer,
            scheduler=scheduler, image_size=image_size, num_frames=num_frames,
            num_inference_steps=int(i_cfg["num_inference_steps"]),
            cfg_scale=float(job["cfg_scale"]), device=device, seed=seed,
            progress_callback=_cb,
        )

        _update_job(job_id, progress=0.92, message="encoding video")
        out_path = _ROOT / "static" / "videos" / f"{job_id}.mp4"
        arr = frames[0].detach().cpu().numpy()
        export_video(arr, out_path, fps=int(job["fps"]),
                     bitrate=str(i_cfg.get("bitrate", "4M")))
        _update_job(job_id, status="completed", progress=1.0, message="done",
                    video_url=f"/video/{job_id}.mp4")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[app] job {job_id} failed:\n{tb}")
        _update_job(job_id, status="failed", message=str(exc), error=tb)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    try:
        cfg_scale = float(data.get("cfg_scale", 7.5))
        num_frames = int(data.get("num_frames", 8))
        image_size = int(data.get("image_size", 64))
        fps = int(data.get("fps", 8))
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid numeric parameters"}), 400
    if not (1.0 <= cfg_scale <= 20.0):
        return jsonify({"error": "cfg_scale must be in [1.0, 20.0]"}), 400
    if not (8 <= num_frames <= 1024):
        return jsonify({"error": "num_frames must be in [8, 1024]"}), 400
    if image_size not in (64, 128, 256):
        return jsonify({"error": "image_size must be one of 64, 128, 256"}), 400
    if not (1 <= fps <= 60):
        return jsonify({"error": "fps must be in [1, 60]"}), 400
    job_id = _new_job(prompt, cfg_scale, num_frames, image_size, fps, seed)
    threading.Thread(target=_run_generation, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/progress", methods=["GET"])
def progress():
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "job_id": job["id"], "status": job["status"], "progress": job["progress"],
        "message": job["message"], "video_url": job["video_url"], "error": job["error"],
    })


@app.route("/video", methods=["GET"])
def video_latest():
    videos_dir = _ROOT / "static" / "videos"
    if not videos_dir.exists():
        return jsonify({"error": "no videos yet"}), 404
    mp4s = sorted(videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        return jsonify({"error": "no videos yet"}), 404
    return send_from_directory(videos_dir, mp4s[0].name)


@app.route("/video/<path:filename>", methods=["GET"])
def video_file(filename: str):
    return send_from_directory(_ROOT / "static" / "videos", filename)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
