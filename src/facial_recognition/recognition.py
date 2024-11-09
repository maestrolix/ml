from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from insightface.model_zoo import ArcFaceONNX
from insightface.utils.face_align import norm_crop
from numpy.typing import NDArray
from PIL import Image

from src.base import InferenceModel
from src.transforms import decode_cv2
from src.schemas import FaceDetectionOutput, FacialRecognitionOutput
from src.session import OrtSession


class FaceRecognizer(InferenceModel):
    def _load(self) -> OrtSession:
        session = self._make_session(self.model_path)
        self.model = ArcFaceONNX(
            self.model_path,
            session=session,
        )
        return session

    def _predict(
        self, inputs: NDArray[np.uint8] | bytes | Image.Image, faces: FaceDetectionOutput, **kwargs: Any
    ) -> FacialRecognitionOutput:
        if faces["boxes"].shape[0] == 0:
            return []
        inputs = decode_cv2(inputs)
        cropped_faces = self._crop(inputs, faces)
        embeddings = self._predict_batch(cropped_faces)
        return self.postprocess(faces, embeddings)

    def _predict_batch(self, cropped_faces: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        embeddings: list[NDArray[np.float32]] = []
        for cropped_face in cropped_faces:
            embeddings.append(self.model.get_feat([cropped_face]))
        return np.concatenate(embeddings, axis=0)

    def postprocess(self, faces: FaceDetectionOutput, embeddings: NDArray[np.float32]) -> FacialRecognitionOutput:
        return [
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "embedding": embedding,
                "score": score,
            }
            for (x1, y1, x2, y2), embedding, score in zip(faces["boxes"], embeddings, faces["scores"])
        ]

    def _crop(self, image: NDArray[np.uint8], faces: FaceDetectionOutput) -> list[NDArray[np.uint8]]:
        return [norm_crop(image, landmark) for landmark in faces["landmarks"]]
