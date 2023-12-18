import io
import csv
import cv2
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends
)
from fastapi.responses import Response, StreamingResponse
import numpy as np
from PIL import Image
from predictor import CursoPredictor
from datetime import datetime as dt
import time

processed_image = None
processed_image_info = None

app = FastAPI(title="Curso reconocedor")

predictor = CursoPredictor()

#face_detector = FaceDetector()

def get_predictor():
    return predictor


def predict_uploadfile(predictor, file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="No es una imagen"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array), img_array

@app.get("/status")
def get_status():
    model_info = {
        "model_name": "Curso Predictor",
        "version": "1.0",
        "status": "en linea",
        "author": "Pablo Badani"
    }

    service_info = {
        "service_name": "Curso reconocedor API",
        "status": "online"
    }

    return {
        "model_info": model_info,
        "service_info": service_info
    }

@app.post("/annotate", responses={
    200: {"content": {"image/jpeg": {}}}
})
def predict_and_annotate(
    file: UploadFile = File(...), 
    predictor: CursoPredictor = Depends(get_predictor)
) -> Response:
    global processed_image
    global processed_image_info
    results, img = predict_uploadfile(predictor, file)
    processed_image_info = {
        "file_name": file.filename,
        "results": results,
        "current_datetime": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": None,
        "model": "Curso Predictor"
    }
    processed_image = img
    new_img = cv2.putText(
        img,
        f"{results['class']} - Confidence: {results['confidence']:.2f}%",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    img_pil = Image.fromarray(new_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)