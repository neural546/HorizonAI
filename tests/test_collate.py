import torch

from AI.collate import collate_fn, random_prompt_dropout


def _item(t=8, c=3, h=64, w=64, length=16):
    return {"frames": torch.randn(t, c, h, w),
            "input_ids": torch.randint(0, 100, (length,)),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "prompt": "a dragon"}


def test_collate_shapes():
    out = collate_fn([_item(), _item()])
    assert out["frames"].shape == (2, 8, 3, 64, 64)
    assert out["input_ids"].shape == (2, 16)
    assert out["attention_mask"].shape == (2, 16)
    assert out["prompt"] == ["a dragon", "a dragon"]


def test_collate_empty_raises():
    try:
        collate_fn([])
    except ValueError:
        return
    assert False, "expected ValueError"


def test_prompt_dropout_shape():
    ids = torch.zeros(4, 8, dtype=torch.long)
    mask = torch.ones(4, 8, dtype=torch.long)
    gen = torch.Generator(); gen.manual_seed(0)
    out = random_prompt_dropout(ids, mask, pad_id=0, prob=0.5, generator=gen)
    assert out.shape == (4,) and out.dtype == torch.float32


def test_prompt_dropout_zero():
    ids = torch.zeros(4, 8, dtype=torch.long)
    mask = torch.ones(4, 8, dtype=torch.long)
    out = random_prompt_dropout(ids, mask, pad_id=0, prob=0.0)
    assert torch.all(out == 0)
