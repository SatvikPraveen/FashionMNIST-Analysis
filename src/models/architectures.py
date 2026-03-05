"""
Model architecture definitions for FashionMNIST-Analysis.

This module contains three CNN architectures of increasing complexity:
    - MiniCNN:   ~106K parameters. Simple two-block baseline.
    - TinyVGG:   ~125K parameters. VGG-style double-conv blocks.
    - ResNet:    ~6.5M parameters. Residual network with skip connections.

All models accept grayscale (1-channel) input and output logits over
the 10 FashionMNIST classes.

Usage:
    from src.models.architectures import MiniCNN, TinyVGG, ResNet, BasicBlock, create_resnet

    model = MiniCNN(in_channels=1, num_classes=10)
    model = TinyVGG(in_channels=1, hidden_units=32, num_classes=10)
    model = create_resnet(num_classes=10)
"""

import torch
from torch import nn


# ============================================================================
# 1. MiniCNN
# ============================================================================

class MiniCNN(nn.Module):
    """
    Lightweight two-block CNN for fast baseline training on Fashion MNIST.

    Architecture:
        Conv(16) → ReLU → MaxPool(2) →
        Conv(32) → ReLU → MaxPool(2) →
        Flatten → Linear(64) → ReLU → Linear(num_classes)

    Input shape:  (N, in_channels, 28, 28)
    Output shape: (N, num_classes)  [raw logits]

    Args:
        in_channels (int): Number of input channels (1 for grayscale).
        num_classes (int): Number of output classes (10 for FashionMNIST).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Two conv blocks halve spatial dims from 28 → 14 → 7
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 → 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14 → 7
        )
        # After pooling: 32 channels × 7 × 7 = 1568 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv blocks then classifier."""
        x = self.conv_block(x)
        return self.classifier(x)


# ============================================================================
# 2. TinyVGG
# ============================================================================

class TinyVGG(nn.Module):
    """
    VGG-inspired architecture with two double-conv blocks for Fashion MNIST.

    Each block uses two 3×3 convolutions followed by MaxPool, mirroring
    the VGG design principle of increased depth before downsampling.

    Architecture:
        [Conv(h) → ReLU → Conv(h) → ReLU → MaxPool(2)] ×2 →
        Flatten → Linear(num_classes)

    Input shape:  (N, in_channels, 28, 28)
    Output shape: (N, num_classes)  [raw logits]

    Args:
        in_channels (int):  Number of input channels (1 for grayscale).
        hidden_units (int): Number of filters in each conv layer.
        num_classes (int):  Number of output classes.
    """

    def __init__(self, in_channels: int, hidden_units: int, num_classes: int):
        super().__init__()
        # Block 1: 28x28 → 14x14
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Block 2: 14x14 → 7x7
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # After two MaxPools: hidden_units × 7 × 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both conv blocks then classifier."""
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


# ============================================================================
# 3. ResNet (BasicBlock + ResNet)
# ============================================================================

class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock with a residual (skip) connection.

    Structure:
        Conv3×3 → BN → ReLU → Conv3×3 → BN
        + downsample(x) if shapes differ
        → ReLU

    The downsampling projection (1×1 conv + BN) is applied automatically
    when the spatial dimensions or channel count change between input and
    output.

    Args:
        in_channels (int):   Number of input feature channels.
        out_channels (int):  Number of output feature channels.
        stride (int):        Stride for the first conv; defaults to 1.
        downsample (nn.Module | None): Optional projection for the skip path.
    """

    expansion = 1  # Channel multiplier (1 for BasicBlock, 4 for Bottleneck)

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Projection shortcut – only used when dimensions do not match
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward pass: F(x) + identity."""
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Project identity if spatial size or channels changed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """
    Flexible ResNet for 28×28 grayscale Fashion MNIST images.

    Adapted from the original paper (He et al., 2016) with a smaller
    initial 3×3 conv and MaxPool instead of the standard 7×7/stride-2
    stem, which is better suited for 28×28 inputs.

    Architecture:
        Conv3×3(64) → BN → ReLU → MaxPool(2) →
        Layer1(64)  → Layer2(128, stride=2) →
        Layer3(256, stride=2) → Layer4(512, stride=2) →
        AdaptiveAvgPool(1×1) → Flatten → Linear(num_classes)

    Args:
        block (nn.Module): Block type to use (e.g. BasicBlock).
        layers (list[int]): Number of blocks in each of the 4 stages.
            Default ResNet-18 config: [2, 2, 2, 2].
        num_classes (int): Number of output classes (default 10).
    """

    def __init__(self, block, layers: list, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        # Stem: one 3×3 conv instead of the 7×7 used for ImageNet
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 28 → 14

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])                  # 14×14
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)        # 7×7
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)        # 4×4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)        # 2×2

        # Pool to 1×1 regardless of input size, then classify
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Construct one residual stage.

        Args:
            block:        Block class (e.g. BasicBlock).
            out_channels: Output channels for every block in this stage.
            num_blocks:   Number of residual blocks to stack.
            stride:       Stride of the first block (used for downsampling between stages).

        Returns:
            nn.Sequential: The assembled stage.
        """
        # Build downsampling projection if channel count or stride changes
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        # First block may downsample; remaining blocks keep the same size
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through stem, residual stages, and classifier."""
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling → flatten → classify
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================================
# Factory helpers
# ============================================================================

def create_resnet(num_classes: int = 10, layers: list = None) -> ResNet:
    """
    Convenience factory for a ResNet-18 style model.

    Args:
        num_classes (int):  Number of output classes. Defaults to 10.
        layers (list[int]): Blocks per stage. Defaults to [2, 2, 2, 2] (ResNet-18).

    Returns:
        ResNet: Instantiated model (un-trained).
    """
    if layers is None:
        layers = [2, 2, 2, 2]
    return ResNet(BasicBlock, layers, num_classes)