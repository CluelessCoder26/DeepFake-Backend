from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File
)

import shutil

from api.predictor import (
    predict_image,
    predict_video
)
from fastapi.staticfiles import (
    StaticFiles
)

from api.gradcam_predictor import (
    generate_gradcam
)
app = FastAPI(
    title="DeepFake Detection API"
)
app.mount(

    "/outputs",

    StaticFiles(
        directory="outputs"
    ),

    name="outputs"
)

UPLOAD_DIR = "uploads"

Path(
    UPLOAD_DIR
).mkdir(
    exist_ok=True
)


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict-image")
async def predict_image_api(
    file: UploadFile = File(...)
):

    file_path = (
        f"{UPLOAD_DIR}/"
        f"{file.filename}"
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_image(
        file_path
    )

    return result
@app.post(
    "/predict-video"
)
async def predict_video_api(

    file: UploadFile = File(...)

):

    file_path = (
        f"{UPLOAD_DIR}/"
        f"{file.filename}"
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_video(
        file_path
    )

    return result
@app.post(
    "/gradcam"
)
async def gradcam_api(

    file: UploadFile = File(...)

):

    file_path = (
        f"{UPLOAD_DIR}/"
        f"{file.filename}"
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = generate_gradcam(
        file_path
    )

    return result