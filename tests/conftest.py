"""Shared pytest fixtures.

All tests in this package are pure logic/contract tests: they exercise
constructor argument handling, pre/post-processing math and cross-class
contracts (e.g. `PoseTracker` <-> detector attribute names). None of them
need a real ONNX Runtime/OpenCV/OpenVINO session or a downloaded model, so
`BaseTool.__init__` is monkeypatched to a lightweight stand-in for every
test, keeping the suite fast, offline and deterministic.
"""
from typing import Optional, Tuple

import pytest

from rtmlib.tools.base import BaseTool


def _fake_base_tool_init(self,
                         onnx_model: Optional[str] = None,
                         model_input_size: Optional[Tuple[int, int]] = None,
                         mean: Optional[tuple] = None,
                         std: Optional[tuple] = None,
                         backend: str = 'opencv',
                         device: str = 'cpu'):
    # Deliberately skip: os.path.exists / download_checkpoint / creating a
    # real cv2.dnn / onnxruntime / openvino session.
    self.onnx_model = onnx_model
    self.model_input_size = model_input_size
    self.mean = mean
    self.std = std
    self.backend = backend
    self.device = device


@pytest.fixture(autouse=True)
def mock_base_tool_init(monkeypatch):
    monkeypatch.setattr(BaseTool, '__init__', _fake_base_tool_init)
