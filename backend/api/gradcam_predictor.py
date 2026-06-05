import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from retinaface import RetinaFace

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from training.model import DeepFakeHybridModel


MODEL_PATH = (
    ROOT_DIR /
    "models" /
    "best_model.pth"
)

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
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


def detect_face(image_path):

    try:

        faces = RetinaFace.detect_faces(
            image_path
        )

        if not isinstance(
            faces,
            dict
        ):
            return None

        if len(faces) == 0:
            return None

        first_face = next(
            iter(faces.values())
        )

        x1, y1, x2, y2 = (
            first_face["facial_area"]
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            return None

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        face = image[
            y1:y2,
            x1:x2
        ]

        return face

    except:

        return None


def generate_gradcam(image_path):

    face = detect_face(
        image_path
    )

    if face is None:

        return {
            "error":
            "No face detected"
        }

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        face_rgb
    )

    image = image.resize(
        (224, 224)
    )

    rgb_img = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    prediction = pred.item()

    labels = {

        0: "FAKE",

        1: "REAL"
    }

    target_layers = [

        model.cnn_branch.conv_head
    ]

    cam = GradCAM(

        model=model,

        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(

        rgb_img,

        grayscale_cam,

        use_rgb=True
    )

    output_img = cv2.cvtColor(

        visualization,

        cv2.COLOR_RGB2BGR
    )

    label_text = (
        f"{labels[prediction]} "
        f"({confidence.item()*100:.2f}%)"
    )

    cv2.putText(

        output_img,

        label_text,

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )

    output_path = (
        OUTPUT_DIR /
        "gradcam_output.jpg"
    )

    cv2.imwrite(
        str(output_path),
        output_img
    )

    return {

        "prediction":
        labels[prediction],

        "confidence":
        round(
            confidence.item() * 100,
            2
        ),

        "gradcam_image":
        "/outputs/gradcam_output.jpg"
    }