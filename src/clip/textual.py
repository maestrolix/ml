import json
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding, Tokenizer

from src.base import InferenceModel
from src.transforms import clean_text
from src.session import OrtSession


class CLIPTextualEncoder(InferenceModel):

    def _predict(self, inputs: str, **kwargs: Any) -> NDArray[np.float32]:
        res: NDArray[np.float32] = self.session.run(None, self.tokenize(inputs))[0][0]
        return res

    def _load(self) -> OrtSession:
        session = super()._load()
        self.tokenizer = self._load_tokenizer()
        tokenizer_kwargs: dict[str, Any] | None = self.text_cfg.get("tokenizer_kwargs")
        self.canonicalize = tokenizer_kwargs is not None and tokenizer_kwargs.get("clean") == "canonicalize"

        return session

    def _load_tokenizer(self) -> Tokenizer:
        context_length: int = self.text_cfg.get("context_length", 77)
        pad_token: str = self.tokenizer_cfg["pad_token"]

        tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_file_path.as_posix())

        pad_id: int = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(length=context_length, pad_token=pad_token, pad_id=pad_id)
        tokenizer.enable_truncation(max_length=context_length)

        return tokenizer

    def tokenize(self, text: str) -> dict[str, NDArray[np.int32]]:
        text = clean_text(text, canonicalize=self.canonicalize)
        tokens: Encoding = self.tokenizer.encode(text)
        return {"text": np.array([tokens.ids], dtype=np.int32)}

    @property
    def model_cfg_path(self) -> Path:
        return Path(self.model_dir).parent / "config.json"

    @property
    def tokenizer_file_path(self) -> Path:
        return Path(self.model_dir) / "tokenizer.json"

    @property
    def tokenizer_cfg_path(self) -> Path:
        return Path(self.model_dir) / "tokenizer_config.json"

    @cached_property
    def model_cfg(self) -> dict[str, Any]:
        model_cfg: dict[str, Any] = json.load(self.model_cfg_path.open())
        return model_cfg

    @property
    def text_cfg(self) -> dict[str, Any]:
        text_cfg: dict[str, Any] = self.model_cfg["text_cfg"]
        return text_cfg

    @cached_property
    def tokenizer_file(self) -> dict[str, Any]:
        tokenizer_file: dict[str, Any] = json.load(self.tokenizer_file_path.open())
        return tokenizer_file

    @cached_property
    def tokenizer_cfg(self) -> dict[str, Any]:
        tokenizer_cfg: dict[str, Any] = json.load(self.tokenizer_cfg_path.open())
        return tokenizer_cfg
