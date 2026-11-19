import shutil
from pathlib import Path

import numpy as np
import pytest

from AI.export import export_video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mp4_export(tmp_path):
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) + i * 20 for i in range(4)]
    path = export_video(frames, tmp_path / "smoke.mp4", fps=4)
    assert Path(path).exists() and Path(path).stat().st_size > 0
