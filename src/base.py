from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        return self._predict(*inputs, **model_kwargs)

    @abstractmethod
    def _predict(self, *inputs: Any, **model_kwargs: Any) -> Any: ...

    def _load(self) -> OrtSession:
        return self._make_session(self.model_path)

    def _make_session(self, model_path: Path) -> OrtSession:
        return OrtSession(model_path)

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir + self.model_name + ".onnx")
