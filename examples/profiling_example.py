#!/usr/bin/env python3
"""
NeuroScope v0.2.0 - Performance Profiling Example

This example demonstrates how to use NeuroScope's performance profiling
features to identify bottleneck layers in your neural network.

Features demonstrated:
- Execution time tracking per layer
- Memory allocation tracking (CUDA)
- Tensor statistics capture (min/max/mean)
- Heatmap visualization in the frontend

Usage:
    python profiling_example.py
"""

import torch
import torch.nn as nn
import neuroscope


class ResidualBlock(nn.Module):
    """Residual block with bottleneck architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class SimpleCNN(nn.Module):
    """Simple CNN for profiling demonstration."""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Residual stages
        self.stage1 = self._make_stage(64, 256, blocks=3, stride=1)
        self.stage2 = self._make_stage(256, 512, blocks=4, stride=2)
        self.stage3 = self._make_stage(512, 1024, blocks=6, stride=2)
        self.stage4 = self._make_stage(1024, 2048, blocks=3, stride=2)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
    
    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


def main():
    print("=" * 60)
    print("NeuroScope v0.2.0 - Performance Profiling Demo")
    print("=" * 60)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create model
    print("\nCreating ResNet-like model...")
    model = SimpleCNN(num_classes=1000).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Attach NeuroScope with profiling ENABLED
    print("\nAttaching NeuroScope with profiling enabled...")
    neuroscope.attach(
        model,
        enable_profiling=True,      # Track execution time per layer
        track_memory=True,          # Track CUDA memory deltas
        capture_tensor_stats=True,  # Capture min/max/mean of outputs
        capture_gradients=False,    # Disable for forward-only profiling
    )
    
    # Start the visualization server
    print("Starting visualization server...")
    neuroscope.start_server(port=8765, open_browser=True)
    
    print("\n" + "=" * 60)
    print("Open http://localhost:5173 in your browser")
    print("Switch to 'Profiling' view mode (press '2') to see heatmap")
    print("=" * 60)
    
    # Create input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    
    print("\nPress Enter to run inference (or 'q' to quit)...")
    
    iteration = 0
    while True:
        user_input = input(f"\n[{iteration}] Run inference? (Enter/q): ").strip().lower()
        
        if user_input == 'q':
            break
        
        # Warmup
        if iteration == 0 and device.type == "cuda":
            print("  Warming up GPU...")
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
        
        # Run inference with profiling
        print("  Running inference...")
        with torch.no_grad():
            output = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Show output info
        print(f"  Output shape: {output.shape}")
        print(f"  Top-5 predictions: {output.topk(5).indices[0].tolist()}")
        
        iteration += 1
    
    # Cleanup
    neuroscope.detach()
    neuroscope.stop_server()
    print("\nNeuroScope detached. Goodbye!")


if __name__ == "__main__":
    main()
