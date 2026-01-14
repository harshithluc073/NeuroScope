"""
TensorFlow-specific tracer.

This tracer captures the computational graph from TensorFlow models
using Keras layer call hooks.

v0.2.0 Features:
- Performance profiling with execution time per layer

IMPORTANT: All hooks are wrapped in try-catch to NEVER crash training.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from neuroscope.core.graph import (
    ExecutionGraph,
    GraphEdge,
    GraphNode,
    NodeType,
    TensorMetadata,
)
from neuroscope.core.tracer import BaseTracer

# Logger for tracer - errors are logged but never raised
logger = logging.getLogger('neuroscope.tracer.tensorflow')

# Maximum nodes to prevent memory issues
MAX_NODES = 10000


class TensorFlowTracer(BaseTracer):
    """
    Tracer for TensorFlow/Keras models.

    Uses Keras layer call hooks to capture the execution graph.
    Supports Sequential, Functional, and Subclassed Keras models.

    v0.2.0: Added enable_profiling for execution time tracking.

    Example:
        >>> tracer = TensorFlowTracer(enable_profiling=True)
        >>> tracer.attach(model)
        >>> output = model(input)
        >>> graph = tracer.get_graph()
        >>> tracer.detach()
    """

    def __init__(
        self,
        suppress_errors: bool = True,
        enable_profiling: bool = True,
    ) -> None:
        """
        Initialize the TensorFlow tracer.
        
        Args:
            suppress_errors: If True, hook errors are logged but don't crash training.
            enable_profiling: Track execution time per layer (default: True)
        """
        super().__init__()
        self._graph = ExecutionGraph()
        self._model: Any = None
        self._original_calls: dict[int, Any] = {}
        self._layer_depths: dict[int, int] = {}
        self._suppress_errors = suppress_errors
        self._error_count = 0
        self._tensor_sources: dict[int, str] = {}
        
        # v0.2.0 profiling
        self._enable_profiling = enable_profiling

    def attach(self, model: Any) -> None:
        """
        Attach to a TensorFlow/Keras model.

        Args:
            model: A tf.keras.Model instance

        Raises:
            ValueError: If model is not a Keras model
            RuntimeError: If already attached
        """
        if self._is_attached:
            raise RuntimeError("Already attached to a model. Call detach() first.")

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlowTracer. "
                "Install with: pip install tensorflow"
            )

        if not isinstance(model, tf.keras.Model):
            raise ValueError(f"Expected tf.keras.Model, got {type(model)}")

        self._model = model
        self._original_calls = {}
        self._layer_depths = {}
        self._tensor_sources = {}

        # Calculate layer depths
        self._calculate_layer_depths(model)

        # Wrap layer calls
        self._wrap_keras_layers(model)

        self._is_attached = True
        self._graph.metadata = {
            "framework": "tensorflow",
            "model_class": type(model).__name__,
            "attached_at": time.time(),
        }

    def detach(self) -> None:
        """Remove tracing from the model."""
        try:
            import tensorflow as tf
            if self._model is not None and isinstance(self._model, tf.keras.Model):
                for layer in self._model.layers:
                    layer_id = id(layer)
                    if layer_id in self._original_calls:
                        layer.call = self._original_calls[layer_id]
        except Exception as e:
            logger.warning(f"Error during detach: {e}")

        self._original_calls = {}
        self._layer_depths = {}
        self._model = None
        self._is_attached = False

    def get_graph(self) -> ExecutionGraph:
        """Get the current execution graph."""
        return self._graph

    def reset_graph(self) -> None:
        """Clear the execution graph."""
        self._graph.clear()
        self._execution_order = 0
        self._tensor_sources = {}
        self._error_count = 0

    def on_forward_start(self, module: Any, inputs: Any, name: str) -> None:
        """Record start of layer execution (not used for TensorFlow)."""
        pass

    def on_forward_end(
        self, module: Any, inputs: Any, outputs: Any, name: str
    ) -> None:
        """Record end of layer execution (called with default timing)."""
        self._on_forward_end_with_time(module, inputs, outputs, name, 0.0)

    def _on_forward_end_with_time(
        self, module: Any, inputs: Any, outputs: Any, name: str, execution_time_ms: float
    ) -> None:
        """Record end of layer execution with timing data."""
        import tensorflow as tf

        # Skip if we've hit node limit
        if len(self._graph.nodes) >= MAX_NODES:
            return

        node_id = f"node_{name}_{self._execution_order}"
        layer_type = type(module).__name__
        full_type = f"{type(module).__module__}.{layer_type}"

        # Extract tensor info
        input_tensors = self._extract_tensor_metadata(inputs)
        output_tensors = self._extract_tensor_metadata(outputs)

        # Check for NaN/Inf
        has_error = False
        error_message = None
        if self._check_tensor_issues(outputs):
            has_error = True
            error_message = "Output contains NaN or Inf values"

        # Get layer depth
        depth = self._layer_depths.get(id(module), 0)

        node = GraphNode(
            id=node_id,
            label=name,
            module_type=full_type,
            node_type=NodeType(self._classify_module_type(layer_type)),
            depth=depth,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            execution_order=self._execution_order,
            execution_time_ms=execution_time_ms,
            has_error=has_error,
            error_message=error_message,
            extra_info=self._get_layer_info(module),
        )

        self._graph.add_node(node)
        
        # Track tensor sources for edges
        self._update_tensor_sources(outputs, node_id)
        self._create_edges_from_inputs(inputs, node_id)
        
        self._execution_order += 1
        self.broadcast("node_update", node.to_dict())

    def _calculate_layer_depths(self, model: Any) -> None:
        """Calculate depth of each layer in the model hierarchy."""
        import tensorflow as tf
        
        def calculate_depth(layer, depth=0):
            self._layer_depths[id(layer)] = depth
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    calculate_depth(sublayer, depth + 1)
        
        calculate_depth(model)

    def _wrap_keras_layers(self, model: Any) -> None:
        """Wrap all Keras layer call methods for tracing."""
        import tensorflow as tf

        def wrap_layer(layer):
            layer_id = id(layer)
            if layer_id in self._original_calls:
                return  # Already wrapped
                
            original_call = layer.call
            self._original_calls[layer_id] = original_call

            tracer = self  # Capture reference

            def wrapped_call(inputs, *args, **kwargs):
                try:
                    # Capture execution time if profiling enabled
                    start_time = time.perf_counter() if tracer._enable_profiling else 0
                    
                    outputs = original_call(inputs, *args, **kwargs)
                    
                    execution_time_ms = 0.0
                    if tracer._enable_profiling:
                        execution_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    tracer._on_forward_end_with_time(
                        layer, inputs, outputs, layer.name, execution_time_ms
                    )
                    return outputs
                except Exception as e:
                    tracer._handle_hook_error("call", layer.name, e)
                    return original_call(inputs, *args, **kwargs)

            layer.call = wrapped_call

            # Recursively wrap sublayers
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    wrap_layer(sublayer)

        for layer in model.layers:
            wrap_layer(layer)

    def _handle_hook_error(self, hook_type: str, name: str, error: Exception) -> None:
        """Handle errors in hooks without crashing training."""
        self._error_count += 1
        
        if self._suppress_errors:
            if self._error_count <= 5:
                logger.warning(f"NeuroScope TF {hook_type} error in '{name}': {error}")
            elif self._error_count == 6:
                logger.warning("NeuroScope: Suppressing further hook errors...")
        else:
            raise error

    def _extract_tensor_metadata(self, tensors: Any) -> list[TensorMetadata]:
        """Extract metadata from TensorFlow tensors."""
        import tensorflow as tf

        result = []

        if isinstance(tensors, (tf.Tensor, tf.Variable)):
            result.append(self._tensor_to_metadata(tensors))
        elif isinstance(tensors, (tuple, list)):
            for t in tensors:
                result.extend(self._extract_tensor_metadata(t))
        elif isinstance(tensors, dict):
            for t in tensors.values():
                result.extend(self._extract_tensor_metadata(t))

        return result

    def _tensor_to_metadata(self, tensor: Any) -> TensorMetadata:
        """Convert TensorFlow tensor to metadata."""
        import tensorflow as tf

        try:
            if tensor.shape.rank is not None:
                shape = tuple(-1 if d is None else d for d in tensor.shape.as_list())
            else:
                shape = ()
        except Exception:
            shape = ()

        try:
            dtype_name = tensor.dtype.name
        except Exception:
            dtype_name = "unknown"

        return TensorMetadata(
            shape=shape,
            dtype=dtype_name,
            device=getattr(tensor, 'device', 'unknown') or 'cpu',
            requires_grad=isinstance(tensor, tf.Variable),
            memory_bytes=0,  # TF doesn't expose this easily
        )

    def _check_tensor_issues(self, tensors: Any) -> bool:
        """Check for NaN or Inf values in tensors."""
        import tensorflow as tf

        try:
            if isinstance(tensors, (tf.Tensor, tf.Variable)):
                return bool(tf.reduce_any(tf.math.is_nan(tensors)) or 
                           tf.reduce_any(tf.math.is_inf(tensors)))
            elif isinstance(tensors, (tuple, list)):
                return any(self._check_tensor_issues(t) for t in tensors)
            elif isinstance(tensors, dict):
                return any(self._check_tensor_issues(t) for t in tensors.values())
        except Exception:
            pass
        return False

    def _update_tensor_sources(self, outputs: Any, node_id: str) -> None:
        """Track which node produced each output tensor."""
        import tensorflow as tf

        if isinstance(outputs, (tf.Tensor, tf.Variable)):
            self._tensor_sources[id(outputs)] = node_id
        elif isinstance(outputs, (tuple, list)):
            for t in outputs:
                self._update_tensor_sources(t, node_id)
        elif isinstance(outputs, dict):
            for t in outputs.values():
                self._update_tensor_sources(t, node_id)

    def _create_edges_from_inputs(self, inputs: Any, target_node_id: str) -> None:
        """Create edges from input tensor sources to this node."""
        import tensorflow as tf

        def process(tensor: Any, idx: int = 0) -> None:
            if isinstance(tensor, (tf.Tensor, tf.Variable)):
                source_id = self._tensor_sources.get(id(tensor))
                if source_id and source_id != target_node_id:
                    edge = GraphEdge(
                        source_id=source_id,
                        target_id=target_node_id,
                        target_input_idx=idx,
                        tensor_info=self._tensor_to_metadata(tensor),
                    )
                    self._graph.add_edge(edge)
            elif isinstance(tensor, (tuple, list)):
                for i, t in enumerate(tensor):
                    process(t, i)
            elif isinstance(tensor, dict):
                for i, t in enumerate(tensor.values()):
                    process(t, i)

        process(inputs)

    def _get_layer_info(self, layer: Any) -> dict[str, Any]:
        """Extract layer configuration info."""
        info: dict[str, Any] = {}

        try:
            config = layer.get_config()
            for key in ["units", "filters", "kernel_size", "activation", "rate", 
                       "num_heads", "key_dim", "use_bias"]:
                if key in config:
                    info[key] = config[key]
        except Exception:
            pass

        try:
            info["parameters"] = layer.count_params()
        except Exception:
            pass

        return info
