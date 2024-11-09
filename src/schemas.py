import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict
from typing import Union, Any, List, Dict, Protocol

class BoundingBox(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int

class FaceDetectionOutput(TypedDict):
    boxes: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    landmarks: npt.NDArray[np.float32]

class DetectedFace(TypedDict):
    boundingBox: BoundingBox
    embedding: npt.NDArray[np.float32]
    score: float


FacialRecognitionOutput = list[DetectedFace]

class SessionNode(Protocol):
    @property
    def name(self) -> str | None: ...

    @property
    def shape(self) -> tuple[int, ...]: ...
