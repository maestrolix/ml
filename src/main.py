from typing import Dict, Union, Any, Annotated
from fastapi import Body, FastAPI, File, Form, Depends
from fastapi.responses import ORJSONResponse
import cv2
from numpy._core.multiarray import CLIP

from src.schemas import FaceDetectionOutput, FacialRecognitionOutput

from src.facial_recognition.detection import FaceDetector
from src.facial_recognition.recognition import FaceRecognizer

from src.clip.textual import CLIPTextualEncoder
from src.clip.visual import CLIPVisualEncoder
from src.transforms import decode_pil

app = FastAPI()


@app.post("/detecting-faces")
def detecting_faces(image: Annotated[bytes, File()]):
    decoded_image = decode_pil(image)
    faces: FaceDetectionOutput = FaceDetector("model", "models/antelopev2/detection").predict(image)
    faces["boxes"] = faces["boxes"].tolist()
    faces["landmarks"] = faces["landmarks"].tolist()
    faces["scores"] = faces["scores"].tolist()
    return ORJSONResponse(faces)

@app.post("/recognition-faces")
def recognition_faces(image: Annotated[bytes, File()]):
    decoded_image = decode_pil(image)
    faces: FaceDetectionOutput = FaceDetector("model", "models/antelopev2/detection/").predict(image)

    with_embedding: FacialRecognitionOutput = FaceRecognizer("model", "models/antelopev2/recognition/").predict(decoded_image, faces)

    return ORJSONResponse(with_embedding)

@app.post("/clip-textual")
def clip_textual(text: str = Form()):
    embedding = CLIPTextualEncoder("model", "models/nllb-clip-base-siglip__v1/textual/").predict(text).tolist()
    return ORJSONResponse(embedding)

@app.post("/visual-textual")
def clip_visual(image: Annotated[bytes, File()]):
    embedding = CLIPVisualEncoder("model", "models/nllb-clip-base-siglip__v1/visual/").predict(image).tolist()
    return ORJSONResponse(embedding)
