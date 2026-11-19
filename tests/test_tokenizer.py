import numpy as np
from PIL import Image

from AI.dataset import repair_dataset, validate_dataset


def test_validate_valid_sample(sample_dataset):
    rep = validate_dataset(sample_dataset)
    assert rep.total_samples == 1 and rep.invalid_samples == 0
    assert rep.samples[0].num_frames == 8


def test_validate_missing_prompt(tmp_path):
    root = tmp_path / "train"
    sample = root / "bad"
    sample.mkdir(parents=True)
    for i in range(1, 9):
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(sample / f"{i:06d}.png")
    rep = validate_dataset(root)
    assert rep.invalid_samples == 1
    assert any("prompt.txt" in e for e in rep.samples[0].errors)


def test_validate_empty_prompt(sample_dataset):
    (sample_dataset / "sample0001" / "prompt.txt").write_text("   \n")
    assert validate_dataset(sample_dataset).invalid_samples == 1


def test_validate_too_few_frames(tmp_path):
    root = tmp_path / "train"
    sample = root / "few"
    sample.mkdir(parents=True)
    (sample / "prompt.txt").write_text("ok\n")
    for i in range(1, 4):
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(sample / f"{i:06d}.png")
    assert validate_dataset(root).invalid_samples == 1


def test_validate_resolution_mismatch(sample_dataset):
    sample = sample_dataset / "sample0001"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(sample / "000003.png")
    assert validate_dataset(sample_dataset).invalid_samples == 1


def test_validate_gap_detection(sample_dataset):
    (sample_dataset / "sample0001" / "000002.png").unlink()
    assert validate_dataset(sample_dataset).invalid_samples == 1


def test_repair_renumbers(sample_dataset):
    (sample_dataset / "sample0001" / "000002.png").unlink()
    actions = repair_dataset(sample_dataset, remove_invalid=False)
    assert isinstance(actions, dict)
    rep = validate_dataset(sample_dataset)
    if rep.total_samples:
        assert rep.samples[0].num_frames == 7


def test_repair_removes_corrupt(sample_dataset):
    sample = sample_dataset / "sample0001"
    (sample / "000004.png").write_bytes(b"not an image")
    repair_dataset(sample_dataset, remove_invalid=False)
    assert not (sample / "000004.png").exists()
