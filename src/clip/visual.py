
import json
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.base import InferenceModel
from src.transforms import crop_pil, decode_pil, get_pil_resampling, normalize, resize_pil, to_numpy
from src.session import OrtSession


class CLIPVisualEncoder(InferenceModel):

    def _predict(self, inputs: Image.Image | bytes, **kwargs: Any) -> NDArray[np.float32]:
        image = decode_pil(inputs)
        res: NDArray[np.float32] = self.session.run(None, self.transform(image))[0][0]
        return res

    def _load(self) -> OrtSession:
        size: list[int] | int = self.preprocess_cfg["size"]
        self.size = size[0] if isinstance(size, list) else size

        self.resampling = get_pil_resampling(self.preprocess_cfg["interpolation"])
        self.mean = np.array(self.preprocess_cfg["mean"], dtype=np.float32)
        self.std = np.array(self.preprocess_cfg["std"], dtype=np.float32)

        return super()._load()

    def transform(self, image: Image.Image) -> dict[str, NDArray[np.float32]]:
        image = resize_pil(image, self.size)
        image = crop_pil(image, self.size)
        image_np = to_numpy(image)
        image_np = normalize(image_np, self.mean, self.std)
        return {"image": np.expand_dims(image_np.transpose(2, 0, 1), 0)}

    @property
    def model_cfg_path(self) -> Path:
        dir = "/".join(self.model_dir.split('/')[0:-2])
        return Path(dir) / "config.json"

    @property
    def preprocess_cfg_path(self) -> Path:
        return Path(self.model_dir) / "preprocess_cfg.json"

    @cached_property
    def model_cfg(self) -> dict[str, Any]:
        model_cfg: dict[str, Any] = json.load(self.model_cfg_path.open())
        return model_cfg

    @cached_property
    def preprocess_cfg(self) -> dict[str, Any]:
        preprocess_cfg: dict[str, Any] = json.load(self.preprocess_cfg_path.open())
        return preprocess_cfg
