"""Video dataset and validation/repair utilities."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}
MIN_FRAMES = 8
MAX_FRAMES = 1024
_FRAME_RE = re.compile(r"^(\d+)$")


@dataclass
class SampleReport:
    sample_id: str
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    num_frames: int = 0
    resolution: Optional[Tuple[int, int]] = None


@dataclass
class ValidationReport:
    root: str
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    samples: List[SampleReport] = field(default_factory=list)

    def add(self, report: SampleReport) -> None:
        self.samples.append(report)
        self.total_samples += 1
        if report.valid:
            self.valid_samples += 1
        else:
            self.invalid_samples += 1

    def summary(self) -> str:
        lines = [
            "Dataset Summary",
            "===============",
            f"Root:            {self.root}",
            f"Total Samples:   {self.total_samples}",
            f"Valid Samples:   {self.valid_samples}",
            f"Invalid Samples: {self.invalid_samples}",
        ]
        if self.invalid_samples:
            lines.append("")
            lines.append("Error Report")
            lines.append("------------")
            for s in self.samples:
                if not s.valid:
                    lines.append(f"  {s.sample_id}:")
                    for err in s.errors:
                        lines.append(f"    - {err}")
        return "\n".join(lines)


def _discover_frames(sample_dir: Path) -> Tuple[List[Path], List[str]]:
    errors: List[str] = []
    frames: List[Tuple[int, Path]] = []
    seen_stems: Dict[int, Path] = {}

    for p in sorted(sample_dir.iterdir()):
        if not p.is_file() or p.name == "prompt.txt":
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            errors.append(f"unsupported file: {p.name}")
            continue
        if not _FRAME_RE.match(p.stem):
            errors.append(f"non-numeric frame name: {p.name}")
            continue
        num = int(p.stem)
        if num in seen_stems:
            errors.append(f"duplicate frame number {num}: {p.name} and {seen_stems[num].name}")
            continue
        seen_stems[num] = p
        frames.append((num, p))

    frames.sort(key=lambda x: x[0])
    for i, (n, _) in enumerate(frames, start=1):
        if n != i:
            errors.append(f"non-sequential numbering: expected {i:06d}, found {n:06d}")
            break
    return [p for _, p in frames], errors


def _read_image_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def validate_sample(sample_dir: Path) -> SampleReport:
    report = SampleReport(sample_id=sample_dir.name)
    prompt_path = sample_dir / "prompt.txt"
    if not prompt_path.exists():
        report.errors.append("missing prompt.txt")
    else:
        try:
            if not prompt_path.read_text(encoding="utf-8").strip():
                report.errors.append("prompt.txt is empty")
        except OSError as exc:
            report.errors.append(f"cannot read prompt.txt: {exc}")

    frames, num_errors = _discover_frames(sample_dir)
    report.errors.extend(num_errors)
    report.num_frames = len(frames)

    if not frames:
        report.errors.append("no image frames found")
    elif len(frames) < MIN_FRAMES:
        report.errors.append(f"too few frames: {len(frames)} < {MIN_FRAMES}")
    elif len(frames) > MAX_FRAMES:
        report.errors.append(f"too many frames: {len(frames)} > {MAX_FRAMES}")

    resolution: Optional[Tuple[int, int]] = None
    seen_hashes: set = set()
    for p in frames:
        arr = _read_image_rgb(p)
        if arr is None:
            report.errors.append(f"unreadable/corrupt image: {p.name}")
            continue
        h, w = arr.shape[:2]
        if resolution is None:
            resolution = (w, h)
        elif (w, h) != resolution:
            report.errors.append(f"resolution mismatch in {p.name}: {(w, h)} != {resolution}")
        digest = hashlib.sha1(arr.tobytes()).hexdigest()
        if digest in seen_hashes:
            report.errors.append(f"duplicate frame detected: {p.name}")
        else:
            seen_hashes.add(digest)

    report.resolution = resolution
    report.valid = len(report.errors) == 0
    return report


def validate_dataset(root: str | Path) -> ValidationReport:
    root = Path(root)
    report = ValidationReport(root=str(root))
    if not root.exists() or not root.is_dir():
        sr = SampleReport(sample_id=str(root), valid=False)
        sr.errors.append("dataset root does not exist or is not a directory")
        report.add(sr)
        return report
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        report.add(validate_sample(sample_dir))
    return report


def repair_dataset(root: str | Path, remove_invalid: bool = True) -> Dict[str, List[str]]:
    root = Path(root)
    actions: Dict[str, List[str]] = {"removed_files": [], "renamed": [], "removed_samples": []}
    if not root.exists():
        return actions

    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in list(sample_dir.iterdir()):
            if not p.is_file() or p.name == "prompt.txt":
                continue
            if p.suffix.lower() not in SUPPORTED_EXTS:
                actions["removed_files"].append(str(p))
                p.unlink()
                continue
            if _read_image_rgb(p) is None:
                actions["removed_files"].append(str(p))
                p.unlink()

        frames: List[Tuple[int, Path]] = []
        for p in sorted(sample_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and _FRAME_RE.match(p.stem):
                frames.append((int(p.stem), p))
        frames.sort(key=lambda x: x[0])

        for i, (old_num, p) in enumerate(frames, start=1):
            if old_num != i:
                final = sample_dir / f"{i:06d}{p.suffix.lower()}"
                p.rename(final)
                actions["renamed"].append(str(final))

        sr = validate_sample(sample_dir)
        if not sr.valid and remove_invalid:
            has_prompt = (sample_dir / "prompt.txt").exists()
            frame_count = sum(1 for p in sample_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
            if not has_prompt or frame_count == 0:
                shutil.rmtree(sample_dir)
                actions["removed_samples"].append(sample_dir.name)

    return actions


try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore


if _TORCH_AVAILABLE:

    class VideoTextDataset(Dataset):  # type: ignore[misc]
        def __init__(self, root: str | Path, image_size: int = 64, num_frames: int = 8,
                     tokenizer=None, max_length: int = 32, validate: bool = False):
            self.root = Path(root)
            self.image_size = int(image_size)
            self.num_frames = int(num_frames)
            self.tokenizer = tokenizer
            self.max_length = int(max_length)
            self.samples = [d for d in sorted(p for p in self.root.iterdir() if p.is_dir())
                            if (d / "prompt.txt").exists()]
            if validate:
                bad = [d.name for d in self.samples if not validate_sample(d).valid]
                if bad:
                    raise ValueError(f"Invalid samples found: {bad}")

        def __len__(self) -> int:
            return len(self.samples)

        def _load_frames(self, sample_dir: Path):
            frames, _ = _discover_frames(sample_dir)
            if not frames:
                raise ValueError(f"No frames in {sample_dir}")
            n = len(frames)
            idxs = np.clip(np.linspace(0, n - 1, self.num_frames).round().astype(int), 0, n - 1)
            arrs = []
            for i in idxs:
                arr = _read_image_rgb(frames[i])
                if arr is None:
                    raise ValueError(f"Corrupt frame {frames[i]}")
                img = Image.fromarray(arr).resize((self.image_size, self.image_size), Image.BILINEAR)
                arrs.append(np.asarray(img))
            return np.stack(arrs, axis=0)

        def __getitem__(self, idx: int):
            sample_dir = self.samples[idx]
            prompt = (sample_dir / "prompt.txt").read_text(encoding="utf-8").strip()
            video = self._load_frames(sample_dir)
            video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            item = {"frames": video_t, "prompt": prompt}
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(prompt)
                item["input_ids"] = torch.tensor(ids, dtype=torch.long)
                item["attention_mask"] = torch.tensor(
                    self.tokenizer.attention_mask(ids), dtype=torch.long
                )
            return item


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset validation and repair")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", metavar="DIR")
    group.add_argument("--repair", metavar="DIR")
    args = parser.parse_args(argv)
    if args.validate:
        report = validate_dataset(args.validate)
        print(report.summary())
        return 0 if report.invalid_samples == 0 else 1
    actions = repair_dataset(args.repair)
    print("Repair Report")
    print("=============")
    for key, items in actions.items():
        print(f"{key}: {len(items)}")
        for item in items:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
