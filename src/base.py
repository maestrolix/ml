from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Any, ClassVar
from huggingface_hub import snapshot_download

from src.session import OrtSession


class InferenceModel(ABC):

    def __init__(
        self,
        model_name: str,
        model_dir: str,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir

    def load(self) -> None:
        self.session = self._load()
        self.loaded = True

    def predict(self, *inputs: Any, **model_kwargs: Any) -> Any:
        self.load()
        if model_kwargs:
            self.configure(**model_kwargs)
        return self._predict(*inputs, **model_kwargs)

    @abstractmethod
    def _predict(self, *inputs: Any, **model_kwargs: Any) -> Any: ...

    def configure(self, **kwargs: Any) -> None:
        pass

    def _load(self) -> OrtSession:
        print(self.model_path)
        return self._make_session(self.model_path)

    def _make_session(self, model_path: Path | str) -> OrtSession:
        return OrtSession(model_path)

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir + self.model_name + ".onnx")
