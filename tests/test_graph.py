"""Tests for the graph data structures."""

import pytest
import json
from neuroscope.core.graph import (
    TensorMetadata,
    GraphNode,
    GraphEdge,
    ExecutionGraph,
    NodeType,
)


class TestTensorMetadata:
    """Tests for TensorMetadata dataclass."""

    def test_creation(self):
        """Test creating TensorMetadata."""
        meta = TensorMetadata(
            shape=(32, 3, 224, 224),
            dtype="float32",
            device="cuda:0",
            requires_grad=True,
            memory_bytes=1024 * 1024,
        )

        assert meta.shape == (32, 3, 224, 224)
        assert meta.dtype == "float32"
        assert meta.device == "cuda:0"
        assert meta.requires_grad is True

    def test_shape_str(self):
        """Test shape string formatting."""
        meta = TensorMetadata(shape=(1, 512, 7, 7), dtype="float32", device="cpu")
        assert meta.shape_str == "[1, 512, 7, 7]"

    def test_numel(self):
        """Test element count calculation."""
        meta = TensorMetadata(shape=(2, 3, 4), dtype="float32", device="cpu")
        assert meta.numel == 24

    def test_serialization(self):
        """Test to_dict and from_dict."""
        meta = TensorMetadata(
            shape=(32, 10),
            dtype="float16",
            device="cuda:0",
            requires_grad=False,
            memory_bytes=640,
        )

        data = meta.to_dict()
        restored = TensorMetadata.from_dict(data)

        assert restored.shape == meta.shape
        assert restored.dtype == meta.dtype
        assert restored.device == meta.device
        assert restored.memory_bytes == meta.memory_bytes


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_creation(self):
        """Test creating GraphNode."""
        node = GraphNode(
            id="node_0",
            label="conv1",
            module_type="torch.nn.Conv2d",
            node_type=NodeType.CONVOLUTION,
            depth=1,
        )

        assert node.id == "node_0"
        assert node.label == "conv1"
        assert node.node_type == NodeType.CONVOLUTION

    def test_with_tensors(self):
        """Test node with input/output tensors."""
        input_meta = TensorMetadata(shape=(1, 3, 224, 224), dtype="float32", device="cpu")
        output_meta = TensorMetadata(shape=(1, 64, 112, 112), dtype="float32", device="cpu")

        node = GraphNode(
            id="node_1",
            label="layer1.0.conv1",
            module_type="torch.nn.Conv2d",
            input_tensors=[input_meta],
            output_tensors=[output_meta],
        )

        assert len(node.input_tensors) == 1
        assert len(node.output_tensors) == 1
        assert node.input_tensors[0].shape == (1, 3, 224, 224)
        assert node.output_tensors[0].shape == (1, 64, 112, 112)

    def test_error_state(self):
        """Test node with error."""
        node = GraphNode(
            id="node_err",
            label="broken_layer",
            module_type="Custom",
            has_error=True,
            error_message="Shape mismatch: expected [32, 10], got [32, 20]",
        )

        assert node.has_error is True
        assert "Shape mismatch" in node.error_message

    def test_serialization(self):
        """Test node serialization."""
        node = GraphNode(
            id="node_2",
            label="fc",
            module_type="torch.nn.Linear",
            node_type=NodeType.LINEAR,
            execution_order=5,
            extra_info={"in_features": 512, "out_features": 10},
        )

        data = node.to_dict()
        restored = GraphNode.from_dict(data)

        assert restored.id == node.id
        assert restored.node_type == node.node_type
        assert restored.extra_info["in_features"] == 512


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_creation(self):
        """Test creating GraphEdge."""
        edge = GraphEdge(
            source_id="node_0",
            target_id="node_1",
            source_output_idx=0,
            target_input_idx=0,
        )

        assert edge.source_id == "node_0"
        assert edge.target_id == "node_1"

    def test_with_tensor_info(self):
        """Test edge with tensor metadata."""
        tensor = TensorMetadata(shape=(32, 256), dtype="float32", device="cpu")
        edge = GraphEdge(
            source_id="attention",
            target_id="ffn",
            tensor_info=tensor,
        )

        assert edge.tensor_info is not None
        assert edge.tensor_info.shape == (32, 256)

    def test_serialization(self):
        """Test edge serialization."""
        edge = GraphEdge(source_id="a", target_id="b")
        data = edge.to_dict()
        restored = GraphEdge.from_dict(data)

        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id


class TestExecutionGraph:
    """Tests for ExecutionGraph dataclass."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = ExecutionGraph()

        # Add nodes
        node1 = GraphNode(id="input", label="input", module_type="Input")
        node2 = GraphNode(id="conv", label="conv", module_type="Conv2d", parent_id="input")
        node3 = GraphNode(id="relu", label="relu", module_type="ReLU")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        # Add edges
        graph.add_edge(GraphEdge(source_id="input", target_id="conv"))
        graph.add_edge(GraphEdge(source_id="conv", target_id="relu"))

        return graph

    def test_add_node(self, sample_graph):
        """Test adding nodes."""
        assert len(sample_graph) == 3
        assert "conv" in sample_graph.nodes

    def test_add_edge(self, sample_graph):
        """Test adding edges."""
        assert len(sample_graph.edges) == 2

    def test_get_root_nodes(self, sample_graph):
        """Test getting root nodes."""
        roots = sample_graph.get_root_nodes()
        root_ids = [n.id for n in roots]

        assert "input" in root_ids
        assert "relu" in root_ids  # No parent
        assert "conv" not in root_ids  # Has parent

    def test_get_error_nodes(self):
        """Test getting error nodes."""
        graph = ExecutionGraph()
        graph.add_node(GraphNode(id="ok", label="ok", module_type="X"))
        graph.add_node(GraphNode(id="err", label="err", module_type="Y", has_error=True))

        errors = graph.get_error_nodes()
        assert len(errors) == 1
        assert errors[0].id == "err"

    def test_clear(self, sample_graph):
        """Test clearing graph."""
        sample_graph.clear()
        assert len(sample_graph) == 0
        assert len(sample_graph.edges) == 0

    def test_json_serialization(self, sample_graph):
        """Test JSON serialization round-trip."""
        json_str = sample_graph.to_json()
        restored = ExecutionGraph.from_json(json_str)

        assert len(restored) == len(sample_graph)
        assert len(restored.edges) == len(sample_graph.edges)
        assert "conv" in restored.nodes

    def test_dict_serialization(self, sample_graph):
        """Test dict serialization."""
        data = sample_graph.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert "timestamp" in data
