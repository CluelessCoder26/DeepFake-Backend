import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import torch

from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace

from training.model import DeepFakeHybridModel


MODEL_PATH = (
    ROOT_DIR /
    "models" /
    "best_model.pth"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = DeepFakeHybridModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


def predict_image(image_path):

    faces = RetinaFace.detect_faces(
        image_path
    )

    if not isinstance(faces, dict):

        return {
            "error":
            "No face detected"
        }

    first_face = next(
        iter(faces.values())
    )

    x1, y1, x2, y2 = (
        first_face["facial_area"]
    )

    image = cv2.imread(
        image_path
    )

    face = image[
        y1:y2,
        x1:x2
    ]

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        face
    )

    tensor = transform(
        image
    ).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

    confidence, pred = torch.max(
        probs,
        dim=1
    )

    labels = {
        0: "FAKE",
        1: "REAL"
    }

    return {

        "prediction":
        labels[pred.item()],

        "confidence":
        round(
            confidence.item() * 100,
            2
        )
    }

def predict_video(video_path):

    FRAME_SKIP = 30

    cap = cv2.VideoCapture(
        video_path
    )

    fake_frames = 0
    real_frames = 0

    processed_faces = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if frame_id % FRAME_SKIP != 0:
            continue

        try:

            faces = RetinaFace.detect_faces(
                frame
            )

            if not isinstance(
                faces,
                dict
            ):
                continue

            if len(faces) == 0:
                continue

            first_face = next(
                iter(faces.values())
            )

            x1, y1, x2, y2 = (
                first_face["facial_area"]
            )

            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                face_rgb
            )

            tensor = transform(
                image
            ).unsqueeze(0)

            tensor = tensor.to(
                device
            )

            with torch.no_grad():

                output = model(
                    tensor
                )

                probs = torch.softmax(
                    output,
                    dim=1
                )

            probs = probs.cpu().numpy()[0]

            if probs[0] > probs[1]:

                fake_frames += 1

            else:

                real_frames += 1

            processed_faces += 1

        except Exception:

            continue

    cap.release()

    if processed_faces == 0:

        return {

            "error":
            "No faces detected"
        }

    fake_percentage = (
        fake_frames /
        processed_faces
    ) * 100

    real_percentage = (
        real_frames /
        processed_faces
    ) * 100

    if fake_percentage >= 30:

        prediction = "FAKE"

        confidence = fake_percentage

    else:

        prediction = "REAL"

        confidence = real_percentage

    return {

        "prediction":
        prediction,

        "confidence":
        round(
            confidence,
            2
        ),

        "faces_processed":
        processed_faces,

        "fake_frames":
        fake_frames,

        "real_frames":
        real_frames
    }