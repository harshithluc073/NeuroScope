"""
Example: Simple MLP visualization with NeuroScope.

This example demonstrates how to attach NeuroScope to a simple
neural network and visualize its execution graph.
"""

import torch
import torch.nn as nn
import neuroscope
import time


def create_mlp():
    """Create a simple MLP model."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1),
    )


def main():
    print("[NeuroScope] Simple MLP Example")
    print("=" * 40)

    # Create model
    model = create_mlp()
    print(f"[+] Created MLP with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Attach NeuroScope
    tracer = neuroscope.attach(model)
    print("[+] Attached NeuroScope tracer")

    # Start visualization server
    server = neuroscope.start_server(port=8765, open_browser=False)
    print("[+] Server started at ws://localhost:8765")

    # Wait a moment for server to be ready
    time.sleep(1)
    
    print("\n[+] Running forward passes...")
    print("    Open http://localhost:3000 in your browser to see the graph")
    print("    Press Ctrl+C to stop\n")

    try:
        pass_count = 0
        while True:
            # Reset graph for fresh capture
            tracer.reset_graph()
            
            # Run forward pass
            sample_input = torch.randn(32, 784)
            with torch.no_grad():
                output = model(sample_input)
            
            # Get and broadcast the complete graph
            graph = tracer.get_graph()
            
            # Broadcast the full graph update
            if server.client_count > 0:
                server.broadcast("graph_update", graph.to_dict())
                
            pass_count += 1
            if pass_count == 1:
                print(f"[+] First forward pass complete!")
                print(f"    Output shape: {output.shape}")
                print(f"    Captured {len(graph)} nodes")
                print(f"    Connected clients: {server.client_count}")
            
            # Wait before next pass
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n[+] Cleaning up...")
        neuroscope.detach()
        neuroscope.stop_server()
        print("[+] Done!")


if __name__ == "__main__":
    main()
