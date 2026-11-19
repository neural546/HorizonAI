"""Video export via FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


def _to_uint8_frames(frames: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D frames (T,C,H,W), got {frames.shape}")
        arr = frames
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[1] in (1, 3, 4):
            arr = arr.transpose(0, 2, 3, 1)
        return [arr[i] for i in range(arr.shape[0])]
    return [np.asarray(f) for f in frames]


def _ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "FFmpeg not found on PATH. Install it and re-run.\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:  brew install ffmpeg"
        )
    return exe


def export_video(frames: Union[np.ndarray, List[np.ndarray]], output_path: str | Path,
                 fps: int = 8, bitrate: str = "4M",
                 resolution: Optional[tuple[int, int]] = None) -> str:
    _ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_list = _to_uint8_frames(frames)
    if not frame_list:
        raise ValueError("No frames provided for export")

    if resolution is not None:
        rw, rh = resolution
        frame_list = [np.asarray(Image.fromarray(f).resize((rw, rh), Image.BILINEAR))
                      for f in frame_list]

    h, w = frame_list[0].shape[:2]
    w -= (w % 2)
    h -= (h % 2)
    frame_list = [f[:h, :w, :3] for f in frame_list]

    with tempfile.TemporaryDirectory(prefix="t2v_frames_") as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(frame_list):
            Image.fromarray(f).save(tmp / f"frame_{i:06d}.png")

        cmd = [
            shutil.which("ffmpeg"), "-y", "-loglevel", "error",
            "-framerate", str(int(fps)),
            "-i", str(tmp / "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", bitrate, "-movflags", "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.strip()}")

    return str(output_path.resolve())


def frames_from_video(video_path: str | Path) -> List[np.ndarray]:
    exe = _ensure_ffmpeg()
    video_path = Path(video_path)
    with tempfile.TemporaryDirectory(prefix="t2v_read_") as tmp:
        tmp = Path(tmp)
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(video_path),
                        str(tmp / "frame_%06d.png")], check=True)
        return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(tmp.glob("frame_*.png"))]


if __name__ == "__main__":
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) + i * 20 for i in range(8)]
    print("Exported:", export_video(frames, "static/videos/output.mp4", fps=8))
