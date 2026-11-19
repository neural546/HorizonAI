"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session", autouse=True)
def _ensure_dirs():
    for d in ("static/videos", "outputs", "AI/checkpoints", "runs", "tests/data/sample_video"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    sample_dir = ROOT / "tests" / "data" / "sample_video"
    if not (sample_dir / "000001.png").exists():
        for i in range(1, 5):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :] = (i * 30, 100 + i * 20, 200 - i * 20)
            arr[16:48, 16:48] = (255 - i * 30, i * 40, 128)
            Image.fromarray(arr).save(sample_dir / f"{i:06d}.png")
    if not (sample_dir / "prompt.txt").exists():
        (sample_dir / "prompt.txt").write_text("a simple colored test scene\n")


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "train"
    sample = root / "sample0001"
    sample.mkdir(parents=True)
    (sample / "prompt.txt").write_text("a dragon flying above snowy mountains\n")
    for i in range(1, 9):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :] = (i * 20, 100, 200 - i * 10)
        arr[16:48, 16:48] = (255 - i * 10, i * 25, 128)
        Image.fromarray(arr).save(sample / f"{i:06d}.png")
    return root


@pytest.fixture()
def flask_app():
    from app import app
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def flask_client(flask_app):
    return flask_app.test_client()
