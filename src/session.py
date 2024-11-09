from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from src.schemas import SessionNode
from src.config import device_id

SUPPORTED_PROVIDERS = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]

class OrtSession:
    def __init__(
        self,
        model_path: Path | str,
        providers: list[str] | None = None,
        provider_options: list[dict[str, Any]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ):
        self.model_path = Path(model_path)
        self.providers = providers if providers is not None else self._providers_default
        self.provider_options = provider_options if provider_options is not None else self._provider_options_default
        self.session = ort.InferenceSession(
            self.model_path.as_posix(),
            providers=self.providers,
            provider_options=self.provider_options,
        )

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

    @property
    def providers(self) -> list[str]:
        return self._providers

    @providers.setter
    def providers(self, providers: list[str]) -> None:
        self._providers = providers

    @property
    def _providers_default(self) -> list[str]:
        available_providers = set(ort.get_available_providers())
        if (openvino := "OpenVINOExecutionProvider") in available_providers:
            device_ids: list[str] = ort.capi._pybind_state.get_available_openvino_device_ids()

            gpu_devices = [device_id for device_id in device_ids if device_id.startswith("GPU")]
            if not gpu_devices:
                available_providers.remove(openvino)
        return [provider for provider in SUPPORTED_PROVIDERS if provider in available_providers]

    @property
    def provider_options(self) -> list[dict[str, Any]]:
        return self._provider_options

    @provider_options.setter
    def provider_options(self, provider_options: list[dict[str, Any]]) -> None:
        self._provider_options = provider_options

    @property
    def _provider_options_default(self) -> list[dict[str, Any]]:
        provider_options = []
        for provider in self.providers:
            match provider:
                case "CPUExecutionProvider":
                    options = {"arena_extend_strategy": "kSameAsRequested"}
                case "CUDAExecutionProvider":
                    options = {"arena_extend_strategy": "kSameAsRequested", "device_id": device_id}
                case "OpenVINOExecutionProvider":
                    options = {
                        "device_type": f"GPU.{device_id}",
                        "precision": "FP32",
                        "cache_dir": (self.model_path.parent / "openvino").as_posix(),
                    }
                case _:
                    options = {}
            provider_options.append(options)
        return provider_options
