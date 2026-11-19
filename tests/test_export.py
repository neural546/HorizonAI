import shutil
from pathlib import Path

import numpy as np
import pytest

from AI.export import export_video, frames_from_video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_export_creates_mp4(tmp_path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) + i * 30 for i in range(8)]
    path = export_video(frames, tmp_path / "test.mp4", fps=8)
    assert Path(path).exists() and Path(path).stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_export_roundtrip(tmp_path):
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) + i * 40 for i in range(4)]
    out = tmp_path / "roundtrip.mp4"
    export_video(frames, out, fps=4)
    read_back = frames_from_video(out)
    assert len(read_back) >= 1 and read_back[0].shape[2] == 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_export_from_tensor(tmp_path):
    import torch
    t = torch.rand(1, 8, 3, 32, 32) * 2 - 1
    assert Path(export_video(t, tmp_path / "tensor.mp4", fps=8)).exists()


def test_export_empty_raises():
    with pytest.raises(ValueError):
        export_video([], "x.mp4")
