import shutil
import time
from pathlib import Path

import pytest

from AI.lightweight.inference import generate_mvp


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mvp_inference_and_export(tmp_path):
    t0 = time.time()
    path = generate_mvp(prompt="a simple test scene", num_inference_steps=3,
                        seed=0, output=tmp_path / "mvp.mp4")
    assert Path(path).exists() and Path(path).stat().st_size > 0
    assert time.time() - t0 < 120
