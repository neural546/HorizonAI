import time
from AI.lightweight.train import train_mvp


def test_mvp_training_completes_quickly():
    t0 = time.time()
    rc = train_mvp(steps=2, seed=0)
    assert rc == 0
    assert time.time() - t0 < 120
