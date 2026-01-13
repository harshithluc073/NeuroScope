# NeuroScope

**Real-time neural network observability for PyTorch, TensorFlow, and JAX**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)]()

NeuroScope visualizes the execution graph of neural networks in real-time. Attach it to your model, run a forward pass, and see the data flow in your browser.

## Installation

```bash
# With PyTorch
pip install neuroscope[pytorch]

# With TensorFlow
pip install neuroscope[tensorflow]

# With JAX
pip install neuroscope[jax]

# Development install (from source)
git clone https://github.com/your-username/neuroscope.git
cd neuroscope
pip install -e ".[dev,pytorch]"
```

---

## Quick Start (PyTorch)

```python
import torch
import torch.nn as nn
import neuroscope

# 1. Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Attach NeuroScope
neuroscope.attach(model)

# 3. Start the server (opens browser)
neuroscope.start_server()

# 4. Run your model - graph updates in real-time!
x = torch.randn(32, 784)
output = model(x)

# 5. Cleanup when done
neuroscope.detach()
```

---

## Usage Examples

### Basic Usage

```python
import neuroscope

# Attach to any PyTorch model
neuroscope.attach(model)

# Start server (opens http://localhost:8765)
neuroscope.start_server()

# Run forward passes - graph visualizes automatically
for batch in dataloader:
    output = model(batch)
    
# Stop when done
neuroscope.stop()
```

### Using with Training Loop

```python
import neuroscope

# Attach before training
neuroscope.attach(model)
neuroscope.start_server(open_browser=False)  # Don't auto-open

print("Open http://localhost:3000 to view graph")

for epoch in range(10):
    for batch in dataloader:
        # Forward pass is captured
        output = model(batch)
        loss = criterion(output, labels)
        
        # Backward pass works normally
        loss.backward()
        optimizer.step()
        
    # Reset graph each epoch (optional)
    neuroscope.reset_graph()
```

### Manual Server Control

```python
from neuroscope import attach, detach
from neuroscope.core.server import NeuroScopeServer

# Attach tracer
tracer = attach(model)

# Create server with custom settings
server = NeuroScopeServer(
    host="0.0.0.0",      # Allow external connections
    port=9000,           # Custom port
    api_key="secret123", # Require authentication
)
server.start()

# Link tracer to server
tracer.set_broadcast_callback(server.broadcast)

# Run model...
output = model(x)

# Cleanup
server.stop()
detach()
```

### TensorFlow/Keras

```python
import tensorflow as tf
from neuroscope.tracers.tensorflow import TensorFlowTracer

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])

tracer = TensorFlowTracer()
tracer.attach(model)

# Run model - layers are traced
output = model(input_data)

graph = tracer.get_graph()
tracer.detach()
```

### JAX

```python
import jax.numpy as jnp
from neuroscope.tracers.jax import JAXTracer

def forward(params, x):
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    return jnp.dot(x, params['w2']) + params['b2']

tracer = JAXTracer()
traced_fn = tracer.attach(forward)

# Run traced function - jaxpr is captured
output = traced_fn(params, x)

graph = tracer.get_graph()
```

---

## Frontend Setup (Development)

The frontend is a React app. For development:

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000 in your browser.

---

## API Reference

### `neuroscope.attach(model)`
Attach tracer hooks to a model.
- **model**: PyTorch nn.Module
- **Returns**: PyTorchTracer instance

### `neuroscope.detach()`
Remove all hooks from the attached model.

### `neuroscope.start_server(open_browser=True)`
Start the WebSocket server.
- **open_browser**: Auto-open browser (default: True)

### `neuroscope.stop()`
Stop the server and clean up.

### `neuroscope.reset_graph()`
Clear the current execution graph.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEUROSCOPE_API_KEY` | API key for authentication | None (disabled) |
| `NEUROSCOPE_HOST` | Server host | `localhost` |
| `NEUROSCOPE_PORT` | Server port | `8765` |

---

## Keyboard Shortcuts (Frontend)

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Focus search |
| `Ctrl+E` | Export as PNG |
| `Escape` | Clear search |
| `Ctrl+0` | Fit view |

---

## Troubleshooting

### "WebSocket connection failed"
Make sure the Python server is running:
```python
neuroscope.start_server()
```

### "No graph displayed"
Run a forward pass after starting the server:
```python
output = model(input_tensor)
```

### Unicode errors on Windows
This is fixed in v0.1.0+. If you see encoding errors, update to the latest version.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
