from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from src.schemas import SessionNode

class OrtSession:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path.as_posix())

    def get_inputs(self) -> list[SessionNode]:
        inputs: list[SessionNode] = self.session.get_inputs()
        return inputs

    def get_outputs(self) -> list[SessionNode]:
        outputs: list[SessionNode] = self.session.get_outputs()
        return outputs

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        outputs: list[NDArray[np.float32]] = self.session.run(output_names, input_feed, run_options)
        return outputs
