"""
PyTorch-specific tracer using forward hooks.

This tracer attaches to PyTorch nn.Module instances and captures
the execution graph during forward passes using register_forward_hook
and register_forward_pre_hook.

v0.2.0 Features:
- Performance profiling with execution time per layer
- Memory tracking (CUDA memory delta per operation)
- Tensor statistics capture (min/max/mean/std)
- Backward pass gradient capture

IMPORTANT: All hooks are wrapped in try-catch to NEVER crash the training loop.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from collections.abc import Callable

from neuroscope.core.graph import (
    ExecutionGraph,
    GraphEdge,
    GraphNode,
    NodeType,
    TensorMetadata,
    TensorStats,
)
from neuroscope.core.tracer import BaseTracer

# Logger for tracer - errors are logged but never raised
logger = logging.getLogger('neuroscope.tracer')

# Maximum nodes to prevent memory issues with very large models
MAX_NODES = 10000


class PyTorchTracer(BaseTracer):
    """
    Tracer for PyTorch models using forward hooks.

    Automatically attaches hooks to all submodules of a model to
    capture tensor shapes, dtypes, and devices during forward passes.

    v0.2.0 Features:
    - enable_profiling: Track execution time per layer
    - track_memory: Track CUDA memory delta per operation
    - capture_tensor_stats: Capture min/max/mean of outputs
    - capture_gradients: Track gradients during backward pass

    Example:
        >>> tracer = PyTorchTracer(enable_profiling=True)
        >>> tracer.attach(model)
        >>> output = model(input)  # Graph is captured with timing
        >>> graph = tracer.get_graph()
        >>> tracer.detach()
    """

    def __init__(
        self,
        suppress_errors: bool = True,
        enable_profiling: bool = True,
        track_memory: bool = True,
        capture_tensor_stats: bool = False,
        capture_gradients: bool = False,
    ) -> None:
        """
        Initialize the PyTorch tracer.
        
        Args:
            suppress_errors: If True, hook errors are logged but don't crash training.
                           Set to False for debugging.
            enable_profiling: Track execution time per layer (default: True)
            track_memory: Track CUDA memory delta per operation (default: True)
            capture_tensor_stats: Capture min/max/mean of outputs (default: False)
            capture_gradients: Track gradients during backward pass (default: False)
        """
        super().__init__()
        self._hooks: list[Any] = []  # RemovableHandle objects
        self._graph = ExecutionGraph()
        self._model: Any = None
        self._module_to_name: dict[int, str] = {}
        self._name_to_depth: dict[str, int] = {}
        self._name_to_parent: dict[str, str | None] = {}
        self._active_modules: list[str] = []  # Stack for tracking nested calls
        self._tensor_sources: dict[int, str] = {}  # tensor id -> source node id
        self._suppress_errors = suppress_errors
        self._error_count = 0
        
        # v0.2.0 profiling options
        self._enable_profiling = enable_profiling
        self._track_memory = track_memory
        self._capture_tensor_stats = capture_tensor_stats
        self._capture_gradients = capture_gradients
        
        # Timing state - stores start time per module
        self._module_start_times: dict[str, float] = {}
        # Memory state - stores memory before forward pass
        self._module_start_memory: dict[str, int] = {}
        # Node ID lookup for gradient updates
        self._name_to_node_id: dict[str, str] = {}

    def attach(self, model: Any) -> None:
        """
        Attach hooks to a PyTorch model.

        Args:
            model: A torch.nn.Module instance

        Raises:
            ValueError: If model is not a torch.nn.Module
            RuntimeError: If already attached to a model
        """
        if self._is_attached:
            raise RuntimeError("Already attached to a model. Call detach() first.")

        # Import torch here to avoid hard dependency
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for PyTorchTracer. Install with: pip install torch")

        if not isinstance(model, nn.Module):
            raise ValueError(f"Expected torch.nn.Module, got {type(model)}")

        self._model = model
        self._hooks = []
        self._module_to_name = {}
        self._name_to_depth = {}
        self._name_to_parent = {}

        # Build module hierarchy
        for name, module in model.named_modules():
            module_id = id(module)
            self._module_to_name[module_id] = name if name else "root"

            # Calculate depth and parent
            parts = name.split(".") if name else []
            depth = len(parts)
            parent_name = ".".join(parts[:-1]) if len(parts) > 1 else (None if len(parts) == 0 else "root")

            self._name_to_depth[name if name else "root"] = depth
            self._name_to_parent[name if name else "root"] = parent_name

            # Register forward hooks
            pre_hook = module.register_forward_pre_hook(
                self._create_pre_hook(name if name else "root", module)
            )
            post_hook = module.register_forward_hook(
                self._create_post_hook(name if name else "root", module)
            )
            self._hooks.extend([pre_hook, post_hook])
            
            # Register backward hook if gradient capture enabled
            if self._capture_gradients:
                try:
                    backward_hook = module.register_full_backward_hook(
                        self._create_backward_hook(name if name else "root", module)
                    )
                    self._hooks.append(backward_hook)
                except Exception as e:
                    logger.debug(f"Could not register backward hook for {name}: {e}")

        self._is_attached = True
        self._graph.metadata = {
            "framework": "pytorch",
            "model_class": type(model).__name__,
            "attached_at": time.time(),
            "profiling_enabled": self._enable_profiling,
            "memory_tracking": self._track_memory,
            "tensor_stats": self._capture_tensor_stats,
            "gradient_capture": self._capture_gradients,
        }

    def detach(self) -> None:
        """Remove all hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._model = None
        self._module_to_name = {}
        self._is_attached = False

    def get_graph(self) -> ExecutionGraph:
        """Get the current execution graph."""
        return self._graph

    def reset_graph(self) -> None:
        """Clear the execution graph for a new forward pass."""
        self._graph.clear()
        self._execution_order = 0
        self._active_modules = []
        self._tensor_sources = {}
        self._module_start_times = {}
        self._module_start_memory = {}
        self._name_to_node_id = {}

    def on_forward_start(self, module: Any, inputs: Any, name: str) -> None:
        """Record the start of a module's forward pass."""
        import torch
        
        self._active_modules.append(name)
        
        # Capture start time for profiling
        if self._enable_profiling:
            self._module_start_times[name] = time.perf_counter()
        
        # Capture memory before forward pass
        if self._track_memory and torch.cuda.is_available():
            try:
                self._module_start_memory[name] = torch.cuda.memory_allocated()
            except Exception:
                pass

    def on_forward_end(
        self, module: Any, inputs: Any, outputs: Any, name: str
    ) -> None:
        """Record the end of a module's forward pass."""
        import torch

        # Calculate execution time
        execution_time_ms = 0.0
        if self._enable_profiling and name in self._module_start_times:
            execution_time_ms = (time.perf_counter() - self._module_start_times[name]) * 1000
            del self._module_start_times[name]
        
        # Calculate memory delta
        memory_delta_bytes = 0
        if self._track_memory and torch.cuda.is_available() and name in self._module_start_memory:
            try:
                current_memory = torch.cuda.memory_allocated()
                memory_delta_bytes = current_memory - self._module_start_memory[name]
                del self._module_start_memory[name]
            except Exception:
                pass

        # Create node
        node_id = f"node_{name}_{self._execution_order}"
        module_type = type(module).__name__
        full_module_type = f"{type(module).__module__}.{module_type}"

        # Extract tensor metadata
        input_tensors = self._extract_tensor_metadata(inputs)
        output_tensors = self._extract_tensor_metadata(outputs)
        
        # Capture tensor statistics if enabled
        tensor_stats = None
        if self._capture_tensor_stats:
            tensor_stats = self._compute_tensor_stats(outputs)

        # Detect shape mismatches or errors
        has_error = False
        error_message = None

        # Check for NaN/Inf in outputs
        if self._check_tensor_issues(outputs):
            has_error = True
            error_message = "Output contains NaN or Inf values"

        # Create node with profiling data
        node = GraphNode(
            id=node_id,
            label=name if name else "root",
            module_type=full_module_type,
            node_type=NodeType(self._classify_module_type(module_type)),
            depth=self._name_to_depth.get(name, 0),
            parent_id=self._get_parent_node_id(name),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            execution_order=self._execution_order,
            execution_time_ms=execution_time_ms,
            memory_delta_bytes=memory_delta_bytes,
            has_error=has_error,
            error_message=error_message,
            tensor_stats=tensor_stats,
            extra_info=self._get_module_info(module),
        )

        self._graph.add_node(node)
        self._execution_order += 1
        
        # Store node ID for gradient capture lookup
        self._name_to_node_id[name] = node_id

        # Track tensor sources for edge creation
        self._update_tensor_sources(outputs, node_id)

        # Create edges from input tensor sources
        self._create_edges_from_inputs(inputs, node_id)

        # Broadcast update
        self.broadcast("node_update", node.to_dict())

        # Pop from active modules
        if self._active_modules and self._active_modules[-1] == name:
            self._active_modules.pop()

    def _create_pre_hook(self, name: str, module: Any) -> Callable:
        """Create a forward pre-hook function with error suppression."""
        def hook(module: Any, inputs: tuple) -> None:
            try:
                self.on_forward_start(module, inputs, name)
            except Exception as e:
                self._handle_hook_error("pre_hook", name, e)
        return hook

    def _create_post_hook(self, name: str, module: Any) -> Callable:
        """Create a forward hook function with error suppression."""
        def hook(module: Any, inputs: tuple, outputs: Any) -> None:
            try:
                # Skip if we've hit node limit
                if len(self._graph.nodes) >= MAX_NODES:
                    return
                self.on_forward_end(module, inputs, outputs, name)
            except Exception as e:
                self._handle_hook_error("post_hook", name, e)
        return hook

    def _handle_hook_error(self, hook_type: str, name: str, error: Exception) -> None:
        """Handle errors in hooks without crashing training."""
        self._error_count += 1
        
        if self._suppress_errors:
            # Log only first few errors to avoid spam
            if self._error_count <= 5:
                logger.warning(f"NeuroScope {hook_type} error in '{name}': {error}")
            elif self._error_count == 6:
                logger.warning("NeuroScope: Suppressing further hook errors...")
        else:
            raise error

    def _extract_tensor_metadata(self, tensors: Any) -> list[TensorMetadata]:
        """Extract metadata from tensors (handles tuples, lists, dicts)."""
        import torch

        result = []

        if isinstance(tensors, torch.Tensor):
            result.append(self._tensor_to_metadata(tensors))
        elif isinstance(tensors, (tuple, list)):
            for t in tensors:
                result.extend(self._extract_tensor_metadata(t))
        elif isinstance(tensors, dict):
            for t in tensors.values():
                result.extend(self._extract_tensor_metadata(t))

        return result

    def _tensor_to_metadata(self, tensor: Any) -> TensorMetadata:
        """Convert a PyTorch tensor to TensorMetadata."""
        return TensorMetadata(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
            memory_bytes=tensor.element_size() * tensor.numel(),
            is_contiguous=tensor.is_contiguous(),
        )

    def _check_tensor_issues(self, tensors: Any) -> bool:
        """Check for NaN or Inf values in tensors."""
        import torch

        if isinstance(tensors, torch.Tensor):
            return bool(torch.isnan(tensors).any() or torch.isinf(tensors).any())
        elif isinstance(tensors, (tuple, list)):
            return any(self._check_tensor_issues(t) for t in tensors)
        elif isinstance(tensors, dict):
            return any(self._check_tensor_issues(t) for t in tensors.values())
        return False

    def _get_parent_node_id(self, name: str) -> str | None:
        """Get the node ID of the parent module."""
        parent_name = self._name_to_parent.get(name)
        if parent_name is None:
            return None

        # Find the most recent node with that parent name
        for node_id, node in reversed(list(self._graph.nodes.items())):
            if node.label == parent_name:
                return node_id
        return None

    def _get_module_info(self, module: Any) -> dict[str, Any]:
        """Extract additional module information."""
        info: dict[str, Any] = {}

        # Count parameters
        try:
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                info["parameters"] = params
        except Exception:
            pass

        # Get specific layer info
        module_type = type(module).__name__

        if hasattr(module, "in_features"):
            info["in_features"] = module.in_features
        if hasattr(module, "out_features"):
            info["out_features"] = module.out_features
        if hasattr(module, "in_channels"):
            info["in_channels"] = module.in_channels
        if hasattr(module, "out_channels"):
            info["out_channels"] = module.out_channels
        if hasattr(module, "kernel_size"):
            info["kernel_size"] = module.kernel_size
        if hasattr(module, "num_heads"):
            info["num_heads"] = module.num_heads

        return info

    def _update_tensor_sources(self, outputs: Any, node_id: str) -> None:
        """Track which node produced each output tensor."""
        import torch

        if isinstance(outputs, torch.Tensor):
            self._tensor_sources[id(outputs)] = node_id
        elif isinstance(outputs, (tuple, list)):
            for t in outputs:
                self._update_tensor_sources(t, node_id)
        elif isinstance(outputs, dict):
            for t in outputs.values():
                self._update_tensor_sources(t, node_id)

    def _create_edges_from_inputs(self, inputs: Any, target_node_id: str) -> None:
        """Create edges from input tensor sources to this node."""
        import torch

        def process(tensor: Any, idx: int = 0) -> None:
            if isinstance(tensor, torch.Tensor):
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

    def _compute_tensor_stats(self, tensors: Any) -> TensorStats | None:
        """Compute statistical summary of output tensors."""
        import torch
        
        def get_stats(tensor: torch.Tensor) -> TensorStats | None:
            try:
                with torch.no_grad():
                    flat = tensor.float().flatten()
                    
                    # Handle empty tensors
                    if flat.numel() == 0:
                        return None
                    
                    return TensorStats(
                        min_val=float(flat.min().item()),
                        max_val=float(flat.max().item()),
                        mean_val=float(flat.mean().item()),
                        std_val=float(flat.std().item()) if flat.numel() > 1 else 0.0,
                        num_zeros=int((flat == 0).sum().item()),
                        num_nan=int(torch.isnan(flat).sum().item()),
                        num_inf=int(torch.isinf(flat).sum().item()),
                    )
            except Exception:
                return None
        
        # Get stats from first tensor found
        if isinstance(tensors, torch.Tensor):
            return get_stats(tensors)
        elif isinstance(tensors, (tuple, list)):
            for t in tensors:
                stats = self._compute_tensor_stats(t)
                if stats:
                    return stats
        elif isinstance(tensors, dict):
            for t in tensors.values():
                stats = self._compute_tensor_stats(t)
                if stats:
                    return stats
        
        return None

    def _create_backward_hook(self, name: str, module: Any) -> Callable:
        """Create a backward hook to capture gradients."""
        def hook(module: Any, grad_input: tuple, grad_output: tuple) -> None:
            try:
                # Find the node for this module
                node_id = self._name_to_node_id.get(name)
                if node_id and node_id in self._graph.nodes:
                    node = self._graph.nodes[node_id]
                    
                    # Extract gradient metadata
                    gradient_tensors = self._extract_tensor_metadata(grad_output)
                    
                    # Update node with gradient info
                    node.gradient_tensors = gradient_tensors
                    
                    # Check for gradient issues
                    if self._check_gradient_issues(grad_output):
                        if not node.has_error:
                            node.has_error = True
                            node.error_message = "Gradient contains NaN, Inf, or vanishing values"
                    
                    # Broadcast gradient update
                    self.broadcast("gradient_update", {
                        "node_id": node_id,
                        "gradient_tensors": [t.to_dict() for t in gradient_tensors],
                    })
            except Exception as e:
                self._handle_hook_error("backward_hook", name, e)
        
        return hook

    def _check_gradient_issues(self, gradients: Any) -> bool:
        """Check for problematic gradients (NaN, Inf, vanishing)."""
        import torch
        
        VANISHING_THRESHOLD = 1e-7
        
        if isinstance(gradients, torch.Tensor):
            if gradients is None:
                return True
            try:
                with torch.no_grad():
                    if torch.isnan(gradients).any():
                        return True
                    if torch.isinf(gradients).any():
                        return True
                    # Check for vanishing gradients
                    if gradients.abs().max().item() < VANISHING_THRESHOLD:
                        return True
            except Exception:
                pass
            return False
        elif isinstance(gradients, (tuple, list)):
            return any(self._check_gradient_issues(g) for g in gradients if g is not None)
        
        return False

