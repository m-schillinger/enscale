import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "enscale"))

from data import build_one_hot


def _cfg(scheme, enabled=True):
    return SimpleNamespace(data={"ensemble_encoding": {"enabled": enabled, "scheme": scheme}})


def test_build_one_hot_gcm_scheme():
    gcm_list = ["G1", "G2", "G1", "G3"]
    rcm_list = ["R1", "R1", "R2", "R2"]

    one_hot = build_one_hot(_cfg("gcm"), gcm_list, rcm_list)

    assert one_hot.shape == (4, 3)
    assert torch.allclose(one_hot[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(one_hot[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(one_hot[3], torch.tensor([0.0, 0.0, 1.0]))


def test_build_one_hot_rcm_scheme():
    gcm_list = ["G1", "G2", "G1", "G3"]
    rcm_list = ["R1", "R1", "R2", "R2"]

    one_hot = build_one_hot(_cfg("rcm"), gcm_list, rcm_list)

    assert one_hot.shape == (4, 2)
    assert torch.allclose(one_hot[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(one_hot[2], torch.tensor([0.0, 1.0]))


def test_build_one_hot_joint_scheme():
    gcm_list = ["G1", "G2"]
    rcm_list = ["R2", "R1"]

    one_hot = build_one_hot(_cfg("gcm+rcm"), gcm_list, rcm_list)

    # Sorted unique order gives gcms=[G1,G2], rcms=[R1,R2]
    assert one_hot.shape == (2, 4)
    assert torch.allclose(one_hot[0], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(one_hot[1], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_build_one_hot_disabled_returns_none():
    gcm_list = ["G1"]
    rcm_list = ["R1"]

    one_hot = build_one_hot(_cfg("gcm+rcm", enabled=False), gcm_list, rcm_list)

    assert one_hot is None
