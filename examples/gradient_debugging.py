#!/usr/bin/env python3
"""
NeuroScope v0.2.0 - Gradient Debugging Example

This example demonstrates how to use NeuroScope's backward pass visualization
to debug vanishing/exploding gradients in deep neural networks.

Features demonstrated:
- Gradient capture during backward pass
- Identifying layers with problematic gradients (NaN, Inf, vanishing)
- Visualizing gradient flow in the frontend

Usage:
    python gradient_debugging.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import neuroscope


# Model with potential gradient issues
class DeepNetwork(nn.Module):
    """A deep network that may suffer from vanishing gradients."""
    
    def __init__(self, num_layers: int = 10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(64, 64))
            # Using sigmoid can cause vanishing gradients in deep networks
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(64, 10))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HealthyNetwork(nn.Module):
    """A network with better gradient flow using ReLU and BatchNorm."""
    
    def __init__(self, num_layers: int = 10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(64, 64))
            layers.append(nn.BatchNorm1d(64))
            layers.append(nn.ReLU())  # Better gradient flow than sigmoid
        layers.append(nn.Linear(64, 10))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_step(model, x, y, optimizer, criterion):
    """Perform one training step."""
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()  # This triggers gradient capture!
    optimizer.step()
    return loss.item()


def main():
    print("=" * 60)
    print("NeuroScope v0.2.0 - Gradient Debugging Demo")
    print("=" * 60)
    
    # Choose which model to debug
    print("\nChoose a model:")
    print("1. DeepNetwork (Sigmoid - may have vanishing gradients)")
    print("2. HealthyNetwork (ReLU + BatchNorm - better gradients)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        model = HealthyNetwork(num_layers=8)
        model_name = "HealthyNetwork"
    else:
        model = DeepNetwork(num_layers=8)
        model_name = "DeepNetwork"
    
    print(f"\nUsing: {model_name}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create sample data
    batch_size = 32
    x = torch.randn(batch_size, 64)
    y = torch.randint(0, 10, (batch_size,))
    
    # Attach NeuroScope with gradient capture ENABLED
    print("\nAttaching NeuroScope with gradient capture...")
    neuroscope.attach(
        model,
        capture_gradients=True,  # v0.2.0: Enable backward pass tracking!
        capture_tensor_stats=True,
    )
    
    # Start the visualization server
    print("Starting visualization server...")
    neuroscope.start_server(port=8765, open_browser=True)
    
    print("\n" + "=" * 60)
    print("Open http://localhost:5173 in your browser")
    print("Switch to 'Gradients' view mode (press '3') to see gradient flow")
    print("=" * 60)
    
    # Training loop
    print("\nStarting training (press Ctrl+C to stop)...")
    
    try:
        for epoch in range(1000):
            loss = train_step(model, x, y, optimizer, criterion)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
            
            # Small delay to allow visualization updates
            import time
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
    
    # Cleanup
    neuroscope.detach()
    neuroscope.stop_server()
    print("\nNeuroScope detached. Goodbye!")


if __name__ == "__main__":
    main()
