import torch.nn as nn
from torchvision import models

NONE_TENS = 10
NUM_TENS_CLASSES = 11
NUM_UNITS_CLASSES = 10


class JerseyNumberNet(nn.Module):
    """Must stay structurally identical to train.py's JerseyNumberNet —
    this is what load_state_dict() expects to match against."""

    def __init__(self, backbone_name: str = "resnet18", dropout: float = 0.3):
        super().__init__()
        if backbone_name not in ("resnet18", "resnet34"):
            raise ValueError("backbone must be resnet18 or resnet34")
        ctor = models.resnet18 if backbone_name == "resnet18" else models.resnet34
        net = ctor(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.visible_head = nn.Linear(feat_dim, 2)
        self.tens_head = nn.Linear(feat_dim, NUM_TENS_CLASSES)
        self.units_head = nn.Linear(feat_dim, NUM_UNITS_CLASSES)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.visible_head(feat), self.tens_head(feat), self.units_head(feat)
