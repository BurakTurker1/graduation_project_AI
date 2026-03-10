from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FAIRFACE_DIR = DATA_DIR / "FairFace"
TRAIN_CSV = FAIRFACE_DIR / "train_labels.csv"
VAL_CSV = FAIRFACE_DIR / "val_labels.csv"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

CASCADE_DIR = ROOT_DIR / "cascades"
FACE_CASCADE = CASCADE_DIR / "haarcascade_frontalface_default.xml"

AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]
GENDER_LABELS = ["Female", "Male"]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_LR = 3e-4
DEFAULT_NUM_WORKERS = 2

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = MODEL_DIR / "age_gender_resnet18.pth"

COCO_LABELS = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

PRODUCT_ALIASES = {
    "cell phone": "phone",
    "tv": "tv",
    "laptop": "laptop",
    "bottle": "bottle",
    "cup": "cup",
    "pizza": "pizza",
    "sandwich": "sandwich",
}


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    video_path: Optional[str] = None
    output_dir: Path = REPORT_DIR
    model_path: Path = DEFAULT_MODEL_PATH
    face_skip: int = 5
    product_skip: int = 10
    product_score: float = 0.6
    enable_products: bool = True
