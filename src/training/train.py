import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    AGE_LABELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_WORKERS,
    FAIRFACE_DIR,
    TRAIN_CSV,
    VAL_CSV,
)
from src.datasets.fairface_dataset import FairFaceDataset
from src.models.age_gender_model import AgeGenderNet
from src.utils.metrics import accuracy
from src.utils.transforms import build_train_transforms, build_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train age/gender model on FairFace")
    parser.add_argument("--data-dir", type=Path, default=FAIRFACE_DIR)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=VAL_CSV)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_age_acc = 0.0
    total_gender_acc = 0.0
    for images, age_targets, gender_targets in loader:
        images = images.to(device)
        age_targets = age_targets.to(device)
        gender_targets = gender_targets.to(device)

        outputs = model(images)
        loss_age = loss_fn(outputs["age"], age_targets)
        loss_gender = loss_fn(outputs["gender"], gender_targets)
        loss = loss_age + loss_gender

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_age_acc += accuracy(outputs["age"], age_targets)
        total_gender_acc += accuracy(outputs["gender"], gender_targets)

    return (
        total_loss / max(1, len(loader)),
        total_age_acc / max(1, len(loader)),
        total_gender_acc / max(1, len(loader)),
    )


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_age_acc = 0.0
    total_gender_acc = 0.0
    with torch.no_grad():
        for images, age_targets, gender_targets in loader:
            images = images.to(device)
            age_targets = age_targets.to(device)
            gender_targets = gender_targets.to(device)

            outputs = model(images)
            loss_age = loss_fn(outputs["age"], age_targets)
            loss_gender = loss_fn(outputs["gender"], gender_targets)
            loss = loss_age + loss_gender

            total_loss += loss.item()
            total_age_acc += accuracy(outputs["age"], age_targets)
            total_gender_acc += accuracy(outputs["gender"], gender_targets)

    return (
        total_loss / max(1, len(loader)),
        total_age_acc / max(1, len(loader)),
        total_gender_acc / max(1, len(loader)),
    )


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    train_set = FairFaceDataset(
        csv_path=args.train_csv,
        root_dir=args.data_dir,
        transform=build_train_transforms(args.image_size),
    )
    val_set = FairFaceDataset(
        csv_path=args.val_csv,
        root_dir=args.data_dir,
        transform=build_val_transforms(args.image_size),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = AgeGenderNet(num_age_classes=len(AGE_LABELS), pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_age_acc, train_gender_acc = train_one_epoch(
        model, train_loader, optimizer, device
        )
        val_loss, val_age_acc, val_gender_acc = evaluate(model, val_loader, device)
        score = (val_age_acc + val_gender_acc) / 2.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_age_acc": train_age_acc,
                "train_gender_acc": train_gender_acc,
                "val_loss": val_loss,
                "val_age_acc": val_age_acc,
                "val_gender_acc": val_gender_acc,
                "score": score,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_age_acc={val_age_acc:.3f} val_gender_acc={val_gender_acc:.3f}"
        )

        if score > best_score:
            best_score = score
            ensure_dir(args.model_out)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "age_labels": AGE_LABELS,
                    "gender_labels": ["Female", "Male"],
                },
                args.model_out,
            )

    if history:
        history_path = args.model_out.with_suffix(".history.csv")
        ensure_dir(history_path)
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print(f"Model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
