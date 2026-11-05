"""Cac mo hinh: SmallCNN (baseline), ResNet50, EfficientNet-B0 (transfer learning)."""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """Baseline: 4 khoi Conv-BN-ReLU-MaxPool + GAP + Dropout + Linear."""
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        chans, layers, c_in = [32, 64, 128, 256], [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, num_classes, freeze_backbone=False, dropout=0.3):
    if name == "small_cnn":
        return SmallCNN(num_classes, dropout)
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        head = m.fc
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
        head = m.classifier
    else:
        raise ValueError(f"Unknown model: {name}")
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m
