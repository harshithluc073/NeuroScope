"""Framework-specific tracers for different ML libraries."""

from neuroscope.tracers.pytorch import PyTorchTracer

__all__ = ["PyTorchTracer"]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "TensorFlowTracer":
        from neuroscope.tracers.tensorflow import TensorFlowTracer
        return TensorFlowTracer
    elif name == "JAXTracer":
        from neuroscope.tracers.jax import JAXTracer
        return JAXTracer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
