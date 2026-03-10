from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import AGE_LABELS, GENDER_LABELS


class FairFaceDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.root_dir / row["file"]
        image = Image.open(image_path).convert("RGB")

        age = row["age"]
        gender = row["gender"]
        try:
            age_label = AGE_LABELS.index(age)
        except ValueError:
            age_label = 0
        try:
            gender_label = GENDER_LABELS.index(gender)
        except ValueError:
            gender_label = 0

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(age_label), torch.tensor(gender_label)
