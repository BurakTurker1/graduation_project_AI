import torch
from torch import nn
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover
    ResNet18_Weights = None


class AgeGenderNet(nn.Module):
    def __init__(self, num_age_classes: int = 9, num_gender_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained and ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT
        try:
            backbone = resnet18(weights=weights)
        except TypeError:
            backbone = resnet18(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.age_head = nn.Linear(in_features, num_age_classes)
        self.gender_head = nn.Linear(in_features, num_gender_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return {"age": age_logits, "gender": gender_logits}
