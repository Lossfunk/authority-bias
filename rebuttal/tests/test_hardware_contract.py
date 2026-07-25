from __future__ import annotations

import pytest

from qwen_rebuttal.model import assert_h200_name


def test_h200_hardware_contract_accepts_h200_variants():
    assert_h200_name("NVIDIA H200")
    assert_h200_name("NVIDIA H200 NVL")


@pytest.mark.parametrize("name", ["NVIDIA H100 80GB HBM3", "NVIDIA A100", "NVIDIA L40S"])
def test_h200_hardware_contract_rejects_other_gpus(name):
    with pytest.raises(RuntimeError, match="H200 is required"):
        assert_h200_name(name)
