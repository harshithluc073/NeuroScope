"""
Abstract base class for framework-specific tracers.

Each supported framework (PyTorch, TensorFlow, JAX) implements
this interface to provide consistent tracing behavior.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from neuroscope.core.graph import ExecutionGraph


class BaseTracer(ABC):
    """
    Abstract base class for neural network tracers.

    Tracers attach to models and capture the execution graph during
    forward passes. They provide a consistent interface across different
    ML frameworks.

    Subclasses must implement the abstract methods to handle
    framework-specific hooking mechanisms.
    """

    # Batch broadcast settings
    BATCH_INTERVAL_MS = 100  # Batch updates every 100ms
    MAX_BATCH_SIZE = 50  # Maximum nodes before forcing broadcast

    def __init__(self) -> None:
        """Initialize the tracer."""
        self._is_attached = False
        self._broadcast_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._execution_order = 0
        
        # Batching state
        self._pending_updates: list[dict[str, Any]] = []
        self._last_broadcast_time = 0.0
        self._batch_lock = threading.Lock()

    @property
    def is_attached(self) -> bool:
        """Whether the tracer is currently attached to a model."""
        return self._is_attached

    def set_broadcast_callback(
        self, callback: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        """
        Set the callback for broadcasting graph updates.

        Args:
            callback: Function that takes (message_type, data) and broadcasts
                     to connected clients. Set to None to disable broadcasting.
        """
        self._broadcast_callback = callback

    def broadcast(self, message_type: str, data: dict[str, Any]) -> None:
        """
        Broadcast a message to connected clients.
        
        For node_update messages, batches updates together to reduce
        WebSocket traffic. Other messages are sent immediately.

        Args:
            message_type: Type of message (e.g., 'graph_update', 'node_update')
            data: Message payload
        """
        if self._broadcast_callback is None:
            return
            
        # For node updates, use batching
        if message_type == "node_update":
            self._queue_node_update(data)
        else:
            # Other messages go immediately
            self._broadcast_callback(message_type, data)

    def _queue_node_update(self, data: dict[str, Any]) -> None:
        """Queue a node update for batched broadcasting."""
        with self._batch_lock:
            self._pending_updates.append(data)
            
            current_time = time.time() * 1000  # ms
            time_since_last = current_time - self._last_broadcast_time
            
            # Flush batch if we have enough updates or enough time has passed
            if (len(self._pending_updates) >= self.MAX_BATCH_SIZE or 
                time_since_last >= self.BATCH_INTERVAL_MS):
                self._flush_batch()

    def _flush_batch(self) -> None:
        """Send all pending updates as a single batch."""
        if not self._pending_updates or self._broadcast_callback is None:
            return
            
        # Send as batch update
        self._broadcast_callback("batch_node_update", {
            "nodes": self._pending_updates,
            "count": len(self._pending_updates),
        })
        
        self._pending_updates = []
        self._last_broadcast_time = time.time() * 1000

    def flush(self) -> None:
        """Force flush any pending batched updates."""
        with self._batch_lock:
            self._flush_batch()

    @abstractmethod
    def attach(self, model: Any) -> None:
        """
        Attach hooks to the model for tracing.

        After calling this method, subsequent forward passes will be
        traced and the execution graph will be captured.

        Args:
            model: The model to attach to (framework-specific type)

        Raises:
            ValueError: If the model type is not supported
            RuntimeError: If already attached to a model
        """
        pass

    @abstractmethod
    def detach(self) -> None:
        """
        Remove all hooks and stop tracing.

        After calling this method, forward passes will no longer be
        traced. Safe to call even if not currently attached.
        """
        pass

    @abstractmethod
    def get_graph(self) -> ExecutionGraph:
        """
        Get the current execution graph.

        Returns:
            The captured execution graph. May be empty if no forward
            pass has been executed since attaching.
        """
        pass

    @abstractmethod
    def reset_graph(self) -> None:
        """
        Clear the current execution graph.

        Call this before a new forward pass if you want to capture
        only that pass's execution.
        """
        pass

    @abstractmethod
    def on_forward_start(self, module: Any, inputs: Any, name: str) -> None:
        """
        Called before a module's forward pass begins.

        Implement this to record the start of a module's execution
        and capture input tensor metadata.

        Args:
            module: The module about to execute
            inputs: Input tensors to the module
            name: The module's name in the model hierarchy
        """
        pass

    @abstractmethod
    def on_forward_end(
        self, module: Any, inputs: Any, outputs: Any, name: str
    ) -> None:
        """
        Called after a module's forward pass completes.

        Implement this to record the completion of a module's execution
        and capture output tensor metadata.

        Args:
            module: The module that just executed
            inputs: Input tensors to the module
            outputs: Output tensors from the module
            name: The module's name in the model hierarchy
        """
        pass

    def _classify_module_type(self, module_class_name: str) -> str:
        """
        Classify a module into a category for styling purposes.

        Args:
            module_class_name: The class name of the module

        Returns:
            A category string (e.g., 'convolution', 'activation', 'attention')
        """
        name_lower = module_class_name.lower()

        if any(k in name_lower for k in ["conv1d", "conv2d", "conv3d", "convolution"]):
            return "convolution"
        elif any(k in name_lower for k in ["linear", "dense", "fc", "matmul"]):
            return "linear"
        elif any(
            k in name_lower
            for k in ["relu", "gelu", "tanh", "sigmoid", "softmax", "activation", "silu", "mish"]
        ):
            return "activation"
        elif any(
            k in name_lower
            for k in ["batchnorm", "layernorm", "groupnorm", "instancenorm", "norm"]
        ):
            return "normalization"
        elif any(k in name_lower for k in ["pool", "avgpool", "maxpool", "adaptivepool"]):
            return "pooling"
        elif any(
            k in name_lower for k in ["attention", "multihead", "selfattention", "crossattention"]
        ):
            return "attention"
        elif any(k in name_lower for k in ["embedding", "embed"]):
            return "embedding"
        elif any(k in name_lower for k in ["dropout"]):
            return "dropout"
        elif any(k in name_lower for k in ["lstm", "gru", "rnn", "recurrent"]):
            return "recurrent"
        elif any(
            k in name_lower
            for k in ["sequential", "modulelist", "moduledict", "container", "block", "layer"]
        ):
            return "container"
        else:
            return "other"
