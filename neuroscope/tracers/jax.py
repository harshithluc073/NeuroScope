"""
JAX-specific tracer using jaxpr introspection.

Captures the computational graph by analyzing the jaxpr
intermediate representation produced by JAX transformations.

IMPORTANT: All tracing is wrapped in try-catch to NEVER crash execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from neuroscope.core.graph import (
    ExecutionGraph,
    GraphEdge,
    GraphNode,
    NodeType,
    TensorMetadata,
)
from neuroscope.core.tracer import BaseTracer

# Logger for tracer
logger = logging.getLogger('neuroscope.tracer.jax')

# Maximum nodes to prevent memory issues
MAX_NODES = 10000


class JAXTracer(BaseTracer):
    """
    Tracer for JAX functions using jaxpr analysis.

    Wraps JAX functions to capture their jaxpr representation
    and convert it to an execution graph. Works with both raw
    JAX functions and Flax/Haiku modules.

    Example:
        >>> tracer = JAXTracer()
        >>> traced_fn = tracer.attach(my_function)
        >>> output = traced_fn(input)
        >>> graph = tracer.get_graph()
        >>> tracer.detach()
    """

    def __init__(self, suppress_errors: bool = True) -> None:
        """
        Initialize the JAX tracer.
        
        Args:
            suppress_errors: If True, tracing errors are logged but don't crash.
        """
        super().__init__()
        self._graph = ExecutionGraph()
        self._original_fn: Callable | None = None
        self._wrapped_fn: Callable | None = None
        self._suppress_errors = suppress_errors
        self._error_count = 0

    def attach(self, model: Any) -> Callable:
        """
        Wrap a JAX function for tracing.

        Unlike PyTorch/TensorFlow, JAX functions are pure so we
        return a wrapped function instead of modifying in-place.

        Args:
            model: A JAX function, Flax module.apply, or Haiku transformed fn

        Returns:
            Wrapped function that captures execution graph

        Raises:
            ValueError: If model is not callable
            RuntimeError: If already attached
        """
        if self._is_attached:
            raise RuntimeError("Already attached. Call detach() first.")

        if not callable(model):
            raise ValueError(f"Expected callable, got {type(model)}")

        try:
            import jax
        except ImportError:
            raise ImportError(
                "JAX is required for JAXTracer. "
                "Install with: pip install jax jaxlib"
            )

        self._original_fn = model
        self._wrapped_fn = self._create_wrapper(model)
        self._is_attached = True

        self._graph.metadata = {
            "framework": "jax",
            "function_name": getattr(model, "__name__", "anonymous"),
            "attached_at": time.time(),
        }

        return self._wrapped_fn

    def detach(self) -> None:
        """Clear the tracer state."""
        self._original_fn = None
        self._wrapped_fn = None
        self._is_attached = False
        self._error_count = 0

    def get_graph(self) -> ExecutionGraph:
        """Get the current execution graph."""
        return self._graph

    def reset_graph(self) -> None:
        """Clear the execution graph."""
        self._graph.clear()
        self._execution_order = 0
        self._error_count = 0

    def on_forward_start(self, module: Any, inputs: Any, name: str) -> None:
        """Not used for JAX tracing."""
        pass

    def on_forward_end(
        self, module: Any, inputs: Any, outputs: Any, name: str
    ) -> None:
        """Not used for JAX tracing."""
        pass

    def _create_wrapper(self, fn: Callable) -> Callable:
        """Create a wrapped function that traces execution."""
        import jax

        tracer = self  # Capture reference

        def wrapper(*args, **kwargs):
            # Reset graph for new execution
            tracer.reset_graph()

            # Get jaxpr representation
            try:
                closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
                tracer._parse_jaxpr(closed_jaxpr)
            except Exception as e:
                tracer._handle_tracing_error(e)

            # Execute the actual function
            return fn(*args, **kwargs)

        return wrapper

    def _handle_tracing_error(self, error: Exception) -> None:
        """Handle tracing errors."""
        self._error_count += 1

        if self._suppress_errors:
            if self._error_count <= 3:
                logger.warning(f"NeuroScope JAX tracing error: {error}")
            self._graph.metadata["tracing_error"] = str(error)
        else:
            raise error

    def _parse_jaxpr(self, closed_jaxpr: Any) -> None:
        """Parse a ClosedJaxpr into graph nodes and edges."""
        import jax

        jaxpr = closed_jaxpr.jaxpr

        # Map variable names to node IDs
        var_to_node: dict[str, str] = {}

        # Add input nodes
        for i, var in enumerate(jaxpr.invars):
            if len(self._graph.nodes) >= MAX_NODES:
                break
                
            node_id = f"input_{i}"
            var_to_node[str(var)] = node_id

            try:
                aval = var.aval
                tensor_meta = self._aval_to_metadata(aval)
            except Exception:
                tensor_meta = TensorMetadata(shape=(), dtype="unknown")

            node = GraphNode(
                id=node_id,
                label=f"Input {i}",
                module_type="jax.input",
                node_type=NodeType.OTHER,
                output_tensors=[tensor_meta],
                execution_order=self._execution_order,
            )
            self._graph.add_node(node)
            self._execution_order += 1

        # Add equation nodes
        for eqn in jaxpr.eqns:
            if len(self._graph.nodes) >= MAX_NODES:
                break
                
            try:
                self._process_equation(eqn, var_to_node)
            except Exception as e:
                if not self._suppress_errors:
                    raise
                logger.debug(f"Error processing equation: {e}")

        # Broadcast complete graph
        self.broadcast("graph_update", self._graph.to_dict())

    def _process_equation(self, eqn: Any, var_to_node: dict[str, str]) -> None:
        """Process a single jaxpr equation into a node."""
        primitive_name = eqn.primitive.name
        node_id = f"op_{self._execution_order}"

        # Get input tensors
        input_tensors = []
        for invar in eqn.invars:
            if hasattr(invar, "aval"):
                input_tensors.append(self._aval_to_metadata(invar.aval))

        # Get output tensors
        output_tensors = []
        for outvar in eqn.outvars:
            if hasattr(outvar, "aval"):
                output_tensors.append(self._aval_to_metadata(outvar.aval))
                var_to_node[str(outvar)] = node_id

        # Extract safe params (skip non-serializable ones)
        extra_info = {}
        if eqn.params:
            for k, v in eqn.params.items():
                try:
                    # Only include serializable params
                    if isinstance(v, (int, float, str, bool, tuple, list)):
                        extra_info[k] = v
                    elif hasattr(v, 'shape'):  # Dimension objects
                        extra_info[k] = str(v)
                except Exception:
                    pass

        node = GraphNode(
            id=node_id,
            label=primitive_name,
            module_type=f"jax.lax.{primitive_name}",
            node_type=NodeType(self._classify_jax_primitive(primitive_name)),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            execution_order=self._execution_order,
            extra_info=extra_info,
        )
        self._graph.add_node(node)

        # Create edges from inputs
        for i, invar in enumerate(eqn.invars):
            source_id = var_to_node.get(str(invar))
            if source_id and source_id != node_id:
                edge = GraphEdge(
                    source_id=source_id,
                    target_id=node_id,
                    target_input_idx=i,
                )
                self._graph.add_edge(edge)

        self._execution_order += 1

    def _aval_to_metadata(self, aval: Any) -> TensorMetadata:
        """Convert JAX abstract value to TensorMetadata."""
        try:
            if hasattr(aval, "shape"):
                shape = tuple(int(d) if hasattr(d, '__int__') else -1 for d in aval.shape)
                dtype = str(aval.dtype) if hasattr(aval, "dtype") else "unknown"
            else:
                shape = ()
                dtype = str(type(aval).__name__)
        except Exception:
            shape = ()
            dtype = "unknown"

        return TensorMetadata(
            shape=shape,
            dtype=dtype,
            device="unknown",
            requires_grad=False,
        )

    def _classify_jax_primitive(self, name: str) -> str:
        """Classify a JAX primitive into a node type."""
        name_lower = name.lower()

        if any(k in name_lower for k in ["conv", "conv_general"]):
            return "convolution"
        elif any(k in name_lower for k in ["dot", "dot_general", "matmul"]):
            return "linear"
        elif any(k in name_lower for k in ["relu", "tanh", "sigmoid", "exp", "log", "gelu", "silu"]):
            return "activation"
        elif any(k in name_lower for k in ["batch_norm", "layer_norm"]):
            return "normalization"
        elif any(k in name_lower for k in ["reduce_window", "reduce_max", "reduce_sum", "max_pool"]):
            return "pooling"
        elif any(k in name_lower for k in ["scatter", "gather", "slice", "concatenate", "reshape"]):
            return "other"
        elif any(k in name_lower for k in ["add", "mul", "sub", "div"]):
            return "other"
        else:
            return "other"
