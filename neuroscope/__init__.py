"""
NeuroScope: Real-time neural network observability tool.

Attach non-intrusive hooks to PyTorch, TensorFlow, or JAX models
to visualize the exact execution graph during forward passes.
"""

from neuroscope.core.graph import ExecutionGraph, GraphEdge, GraphNode, TensorMetadata
from neuroscope.core.server import NeuroScopeServer
from neuroscope.core.tracer import BaseTracer

__version__ = "0.1.0"
__all__ = [
    "attach",
    "detach",
    "start_server",
    "stop_server",
    "ExecutionGraph",
    "GraphNode",
    "GraphEdge",
    "TensorMetadata",
]

# Global state
_active_tracer: BaseTracer | None = None
_server: NeuroScopeServer | None = None


def attach(model, framework: str | None = None) -> BaseTracer:
    """
    Attach NeuroScope tracer to a model.

    Args:
        model: The neural network model (PyTorch, TensorFlow, or JAX)
        framework: Optional framework hint ('pytorch', 'tensorflow', 'jax').
                   If None, auto-detects from model type.

    Returns:
        The attached tracer instance

    Example:
        >>> import neuroscope
        >>> tracer = neuroscope.attach(model)
    """
    global _active_tracer

    if _active_tracer is not None:
        _active_tracer.detach()

    # Auto-detect framework if not specified
    if framework is None:
        framework = _detect_framework(model)

    # Create appropriate tracer
    tracer = _create_tracer(framework)
    tracer.attach(model)
    _active_tracer = tracer

    # Connect to server if running
    if _server is not None:
        tracer.set_broadcast_callback(_server.broadcast)

    return tracer


def detach() -> None:
    """
    Detach the active tracer and remove all hooks.

    Example:
        >>> neuroscope.detach()
    """
    global _active_tracer

    if _active_tracer is not None:
        _active_tracer.detach()
        _active_tracer = None


def start_server(port: int = 8765, open_browser: bool = True) -> NeuroScopeServer:
    """
    Start the NeuroScope visualization server.

    Args:
        port: WebSocket server port (default: 8765)
        open_browser: Whether to open the browser automatically

    Returns:
        The server instance

    Example:
        >>> neuroscope.start_server()  # Opens browser at localhost:8765
    """
    global _server

    if _server is not None:
        _server.stop()

    _server = NeuroScopeServer(port=port)
    _server.start(open_browser=open_browser)

    # Connect existing tracer to server
    if _active_tracer is not None:
        _active_tracer.set_broadcast_callback(_server.broadcast)

    return _server


def stop_server() -> None:
    """
    Stop the NeuroScope visualization server.

    Example:
        >>> neuroscope.stop_server()
    """
    global _server

    if _server is not None:
        _server.stop()
        _server = None


def _detect_framework(model) -> str:
    """Auto-detect the ML framework from the model type."""
    model_type = type(model).__module__

    if "torch" in model_type:
        return "pytorch"
    elif "tensorflow" in model_type or "keras" in model_type:
        return "tensorflow"
    elif "flax" in model_type or "haiku" in model_type:
        return "jax"
    else:
        # Try checking for common base classes
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                return "pytorch"
        except ImportError:
            pass

        try:
            import tensorflow as tf
            if isinstance(model, (tf.Module, tf.keras.Model)):
                return "tensorflow"
        except ImportError:
            pass

        raise ValueError(
            f"Could not detect framework for model type: {type(model)}. "
            "Please specify framework='pytorch', 'tensorflow', or 'jax'"
        )


def _create_tracer(framework: str) -> BaseTracer:
    """Create a tracer instance for the specified framework."""
    if framework == "pytorch":
        from neuroscope.tracers.pytorch import PyTorchTracer
        return PyTorchTracer()
    elif framework == "tensorflow":
        from neuroscope.tracers.tensorflow import TensorFlowTracer
        return TensorFlowTracer()
    elif framework == "jax":
        from neuroscope.tracers.jax import JAXTracer
        return JAXTracer()
    else:
        raise ValueError(f"Unknown framework: {framework}")
