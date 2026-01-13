"""
Example: ResNet visualization with NeuroScope.

This example shows how NeuroScope handles complex hierarchical
models with residual connections.
"""

import torch
import torch.nn as nn
import neuroscope


class ResidualBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet-like architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    print("🔬 NeuroScope - ResNet Example")
    print("=" * 40)

    # Create model
    model = SimpleResNet(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Created ResNet with {param_count:,} parameters")

    # Attach and start server
    neuroscope.attach(model)
    neuroscope.start_server()

    # Run forward pass
    print("\nRunning forward pass with 224x224 image...")
    sample_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(sample_input)

    print(f"✓ Output shape: {output.shape}")

    graph = neuroscope._active_tracer.get_graph()
    print(f"✓ Captured {len(graph)} nodes")

    # Count node types
    type_counts = {}
    for node in graph.nodes.values():
        t = node.node_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\nNode types:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    print("\n" + "=" * 40)
    print("Press Ctrl+C to exit")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up...")
        neuroscope.detach()
        neuroscope.stop_server()


if __name__ == "__main__":
    main()
