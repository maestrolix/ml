from typing import Any

import numpy as np
from insightface.model_zoo import RetinaFace
from numpy.typing import NDArray

from src.base import InferenceModel
from src.transforms import decode_cv2
from src.schemas import FaceDetectionOutput
from src.session import OrtSession

from PIL import Image


class FaceDetector(InferenceModel):

    def _load(self) -> OrtSession:
        session = self._make_session(self.model_path)
        self.model = RetinaFace(session=session)
        self.model.prepare(ctx_id=0, input_size=(640, 640))

        return session

    def _predict(self, inputs: NDArray[np.uint8] | bytes | Image.Image, **kwargs: Any) -> FaceDetectionOutput:
        inputs = decode_cv2(inputs)

        bboxes, landmarks = self._detect(inputs)
        return {
            "boxes": bboxes[:, :4].round(),
            "scores": bboxes[:, 4],
            "landmarks": landmarks,
        }

    def _detect(self, inputs: NDArray[np.uint8] | bytes) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self.model.detect(inputs)  # type: ignore
