"""
Graph data structures for representing neural network execution.

These dataclasses provide a framework-agnostic representation of
the computational graph captured during model execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Categories of neural network operations."""

    CONVOLUTION = "convolution"
    LINEAR = "linear"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    POOLING = "pooling"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    DROPOUT = "dropout"
    RECURRENT = "recurrent"
    CONTAINER = "container"  # Sequential, ModuleList, etc.
    OTHER = "other"


@dataclass
class TensorMetadata:
    """
    Metadata about a tensor at a specific point in the computation.

    Attributes:
        shape: Tensor dimensions as a tuple
        dtype: Data type string (e.g., 'float32', 'int64')
        device: Device location (e.g., 'cpu', 'cuda:0')
        requires_grad: Whether gradients are tracked
        memory_bytes: Estimated memory consumption
        is_contiguous: Whether tensor memory is contiguous
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool = False
    memory_bytes: int = 0
    is_contiguous: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "memory_bytes": self.memory_bytes,
            "is_contiguous": self.is_contiguous,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMetadata:
        """Create from dictionary."""
        return cls(
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            device=data["device"],
            requires_grad=data.get("requires_grad", False),
            memory_bytes=data.get("memory_bytes", 0),
            is_contiguous=data.get("is_contiguous", True),
        )

    @property
    def shape_str(self) -> str:
        """Human-readable shape string like '[32, 3, 224, 224]'."""
        return f"[{', '.join(map(str, self.shape))}]"

    @property
    def numel(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class GraphNode:
    """
    A node in the execution graph representing a module or operation.

    Attributes:
        id: Unique identifier for this node
        label: Human-readable name (e.g., 'conv1', 'layer.0.attention')
        module_type: Full class name (e.g., 'torch.nn.Conv2d')
        node_type: Categorized node type for styling
        depth: Nesting level in module hierarchy (0 = top-level)
        parent_id: ID of parent container node, if any
        children_ids: IDs of child nodes (for containers)
        input_tensors: Metadata for input tensors
        output_tensors: Metadata for output tensors
        execution_order: Order in which this node executed (0-indexed)
        has_error: Whether an error was detected at this node
        error_message: Description of the error, if any
        extra_info: Additional framework-specific metadata
    """

    id: str
    label: str
    module_type: str
    node_type: NodeType = NodeType.OTHER
    depth: int = 0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    input_tensors: list[TensorMetadata] = field(default_factory=list)
    output_tensors: list[TensorMetadata] = field(default_factory=list)
    execution_order: int = 0
    has_error: bool = False
    error_message: str | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "module_type": self.module_type,
            "node_type": self.node_type.value,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "input_tensors": [t.to_dict() for t in self.input_tensors],
            "output_tensors": [t.to_dict() for t in self.output_tensors],
            "execution_order": self.execution_order,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "extra_info": self.extra_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            module_type=data["module_type"],
            node_type=NodeType(data.get("node_type", "other")),
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            input_tensors=[TensorMetadata.from_dict(t) for t in data.get("input_tensors", [])],
            output_tensors=[TensorMetadata.from_dict(t) for t in data.get("output_tensors", [])],
            execution_order=data.get("execution_order", 0),
            has_error=data.get("has_error", False),
            error_message=data.get("error_message"),
            extra_info=data.get("extra_info", {}),
        )


@dataclass
class GraphEdge:
    """
    An edge connecting two nodes in the execution graph.

    Represents data flow from source node's output to target node's input.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        source_output_idx: Index of the output tensor on source node
        target_input_idx: Index of the input tensor on target node
        tensor_info: Metadata about the tensor flowing through this edge
    """

    source_id: str
    target_id: str
    source_output_idx: int = 0
    target_input_idx: int = 0
    tensor_info: TensorMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_output_idx": self.source_output_idx,
            "target_input_idx": self.target_input_idx,
            "tensor_info": self.tensor_info.to_dict() if self.tensor_info else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        """Create from dictionary."""
        tensor_info = None
        if data.get("tensor_info"):
            tensor_info = TensorMetadata.from_dict(data["tensor_info"])
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            source_output_idx=data.get("source_output_idx", 0),
            target_input_idx=data.get("target_input_idx", 0),
            tensor_info=tensor_info,
        )


@dataclass
class ExecutionGraph:
    """
    Complete execution graph captured during a forward pass.

    Attributes:
        nodes: Dictionary mapping node IDs to GraphNode instances
        edges: List of edges connecting nodes
        metadata: Additional information about the graph
        timestamp: When this graph was captured
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_root_nodes(self) -> list[GraphNode]:
        """Get all top-level nodes (no parent)."""
        return [node for node in self.nodes.values() if node.parent_id is None]

    def get_children(self, node_id: str) -> list[GraphNode]:
        """Get all direct children of a node."""
        if node_id not in self.nodes:
            return []
        return [
            self.nodes[child_id]
            for child_id in self.nodes[node_id].children_ids
            if child_id in self.nodes
        ]

    def get_error_nodes(self) -> list[GraphNode]:
        """Get all nodes with errors."""
        return [node for node in self.nodes.values() if node.has_error]

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionGraph:
        """Create from dictionary."""
        graph = cls(
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
        for node_id, node_data in data.get("nodes", {}).items():
            graph.nodes[node_id] = GraphNode.from_dict(node_data)
        for edge_data in data.get("edges", []):
            graph.edges.append(GraphEdge.from_dict(edge_data))
        return graph

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionGraph:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)
