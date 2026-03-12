import argparse
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import torch
from PIL import Image
from torchvision import transforms as T

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    AGE_LABELS,
    COCO_LABELS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_PATH,
    FACE_CASCADE,
    GENDER_LABELS,
    PRODUCT_ALIASES,
    REPORT_DIR,
)
from src.models.age_gender_model import AgeGenderNet
from src.utils.reporting import finalize_report
from src.utils.transforms import build_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam demo for age/gender + product report")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--face-skip", type=int, default=5, help="Run face detection every N frames")
    parser.add_argument("--product-skip", type=int, default=10, help="Run product detection every N frames")
    parser.add_argument("--product-score", type=float, default=0.6)
    parser.add_argument("--no-products", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--dedup-seconds", type=int, default=10)
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    return parser.parse_args()


def load_age_gender_model(model_path: Path, device: str):
    if not model_path.exists():
        print(f"Age/Gender model not found: {model_path}")
        return None, AGE_LABELS, GENDER_LABELS

    model = AgeGenderNet(num_age_classes=len(AGE_LABELS), pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
    age_labels = checkpoint.get("age_labels", AGE_LABELS)
    gender_labels = checkpoint.get("gender_labels", GENDER_LABELS)
    model.to(device)
    model.eval()
    return model, age_labels, gender_labels


def load_product_model(device: str):
    try:
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        model.to(device)
        model.eval()
        return model
    except Exception:
        try:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

            model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
            model.to(device)
            model.eval()
            return model
        except Exception as exc:  # pragma: no cover
            print(f"Product model disabled: {exc}")
            return None


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def detect_faces(cascade, frame, min_size=(40, 40), downscale=0.5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if downscale != 1.0:
        small = cv2.resize(gray, None, fx=downscale, fy=downscale)
        faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        scaled = []
        for (x, y, w, h) in faces:
            scaled.append((int(x / downscale), int(y / downscale), int(w / downscale), int(h / downscale)))
        return scaled
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def predict_age_gender(model, transform, device, face_bgr, age_labels, gender_labels):
    if model is None:
        return "unknown", 0.0, "unknown", 0.0

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        age_probs = torch.softmax(outputs["age"], dim=1)[0]
        gender_probs = torch.softmax(outputs["gender"], dim=1)[0]
    age_idx = int(torch.argmax(age_probs).item())
    gender_idx = int(torch.argmax(gender_probs).item())

    age_label = age_labels[age_idx] if age_idx < len(age_labels) else "unknown"
    gender_label = gender_labels[gender_idx] if gender_idx < len(gender_labels) else "unknown"

    return age_label, float(age_probs[age_idx].item()), gender_label, float(gender_probs[gender_idx].item())


def detect_product(model, frame, device, score_thresh: float):
    if model is None:
        return "unknown", 0.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = T.ToTensor()(rgb).to(device)
    with torch.no_grad():
        outputs = model([tensor])[0]

    labels = outputs.get("labels", [])
    scores = outputs.get("scores", [])

    best_label = "unknown"
    best_score = 0.0

    for label, score in zip(labels, scores):
        score_val = float(score.item())
        if score_val < score_thresh:
            break

        label_idx = int(label)
        if label_idx < 0 or label_idx >= len(COCO_LABELS):
            continue

        label_name = COCO_LABELS[label_idx]
        if label_name == "person":
            continue

        if score_val > best_score:
            best_score = score_val
            best_label = PRODUCT_ALIASES.get(label_name, label_name)

    return best_label, best_score


def main() -> None:
    args = parse_args()

    cascade = cv2.CascadeClassifier(str(FACE_CASCADE))
    if cascade.empty():
        raise RuntimeError(f"Face cascade not found: {FACE_CASCADE}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"
    age_model, age_labels, gender_labels = load_age_gender_model(args.model_path, device)
    transform = build_val_transforms(DEFAULT_IMAGE_SIZE)

    product_model = None
    if not args.no_products:
        product_model = load_product_model(device)

    cap = cv2.VideoCapture(args.camera_index if args.video_path is None else args.video_path)
    if not cap.isOpened():
        raise RuntimeError("Camera/Video could not be opened")

    events: List[Dict[str, str]] = []
    recent: Deque[Dict[str, object]] = deque(maxlen=100)
    last_product = ("unknown", 0.0)

    frame_idx = 0
    faces_cache: List[Tuple[int, int, int, int]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % max(1, args.face_skip) == 0:
            faces_cache = detect_faces(cascade, frame)

        if not args.no_products and frame_idx % max(1, args.product_skip) == 0:
            last_product = detect_product(product_model, frame, device, args.product_score)

        frame_h, frame_w = frame.shape[:2]
        for (x, y, w, h) in faces_cache:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)
            bbox = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            age_label, age_conf, gender_label, gender_conf = predict_age_gender(
                age_model, transform, device, face, age_labels, gender_labels
            )
            timestamp = datetime.now()

            duplicate = False
            for item in list(recent):
                delta = (timestamp - item["timestamp"]).total_seconds()
                if delta <= args.dedup_seconds and iou(item["bbox"], bbox) >= args.dedup_iou:
                    duplicate = True
                    break

            if not duplicate:
                product_label, product_conf = last_product
                events.append(
                    {
                        "timestamp": timestamp.isoformat(timespec="seconds"),
                        "age": age_label,
                        "age_conf": round(age_conf, 3),
                        "gender": gender_label,
                        "gender_conf": round(gender_conf, 3),
                        "product": product_label,
                        "product_conf": round(product_conf, 3),
                    }
                )
                recent.append({"timestamp": timestamp, "bbox": bbox})

            if not args.no_display:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                label = f"{gender_label} {age_label}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 0),
                    2,
                )

        if not args.no_display:
            cv2.putText(
                frame,
                f"Product: {last_product[0]}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 140, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press Q or ESC to Quit",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Customer Analytics Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Kamera kapatılıyor...")
                break

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    paths = finalize_report(events, args.output_dir)
    print(f"Rapor kaydedildi: {paths['events']} ve {paths['summary']}")


if __name__ == "__main__":
    main()