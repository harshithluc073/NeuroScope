"""
Example: Transformer visualization with NeuroScope.

This example demonstrates how NeuroScope handles attention-based
models with self-attention and cross-attention layers.
"""

import torch
import torch.nn as nn
import neuroscope


def main():
    print("🔬 NeuroScope - Transformer Example")
    print("=" * 40)

    # Create a standard Transformer model
    model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Created Transformer with {param_count:,} parameters")
    print(f"  - d_model: 512")
    print(f"  - heads: 8")
    print(f"  - encoder layers: 2")
    print(f"  - decoder layers: 2")

    # Attach and start server
    neuroscope.attach(model)
    neuroscope.start_server()

    # Run forward pass
    print("\nRunning forward pass...")
    batch_size = 2
    seq_len_src = 10
    seq_len_tgt = 8

    src = torch.randn(batch_size, seq_len_src, 512)  # Source sequence
    tgt = torch.randn(batch_size, seq_len_tgt, 512)  # Target sequence

    with torch.no_grad():
        output = model(src, tgt)

    print(f"✓ Source shape: {src.shape}")
    print(f"✓ Target shape: {tgt.shape}")
    print(f"✓ Output shape: {output.shape}")

    graph = neuroscope._active_tracer.get_graph()
    print(f"\n✓ Captured {len(graph)} nodes")

    # Find attention layers
    attention_nodes = [
        n for n in graph.nodes.values()
        if "attention" in n.module_type.lower() or n.node_type.value == "attention"
    ]
    print(f"✓ Found {len(attention_nodes)} attention-related nodes")

    print("\n" + "=" * 40)
    print("View the graph to see:")
    print("  - Self-attention in encoder layers")
    print("  - Masked self-attention in decoder layers")
    print("  - Cross-attention between encoder and decoder")
    print("\nPress Ctrl+C to exit")

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
