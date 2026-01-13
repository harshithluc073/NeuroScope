"""Pytest configuration and shared fixtures."""

import pytest
import torch
import torch.nn as nn


# ===== Model Fixtures =====

@pytest.fixture
def simple_linear():
    """A simple linear layer."""
    return nn.Linear(10, 5)


@pytest.fixture
def simple_mlp():
    """A simple MLP model."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def conv_model():
    """A simple CNN model."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def nested_model():
    """A model with nested containers."""
    class Block(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)
            self.norm = nn.LayerNorm(out_features)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.norm(self.fc(x)))

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = Block(10, 20)
            self.block2 = Block(20, 10)
            self.blocks = nn.ModuleList([Block(10, 10) for _ in range(3)])
            self.final = nn.Linear(10, 5)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            for block in self.blocks:
                x = x + block(x)
            return self.final(x)

    return NestedModel()


@pytest.fixture
def attention_model():
    """A model with attention."""
    return nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)


# ===== Tensor Fixtures =====

@pytest.fixture
def sample_batch():
    """A sample input batch for MLP models."""
    return torch.randn(4, 10)


@pytest.fixture
def sample_image_batch():
    """A sample input batch for CNN models."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def sample_sequence():
    """A sample input for attention models."""
    return torch.randn(2, 8, 64)


# ===== Tracer Fixtures =====

@pytest.fixture
def pytorch_tracer():
    """Create a fresh PyTorch tracer."""
    from neuroscope.tracers.pytorch import PyTorchTracer
    tracer = PyTorchTracer()
    yield tracer
    if tracer.is_attached:
        tracer.detach()


# ===== Server Fixtures =====

@pytest.fixture
def server():
    """Create a fresh server instance."""
    from neuroscope.core.server import NeuroScopeServer
    srv = NeuroScopeServer(port=9998)  # Different port to avoid conflicts
    yield srv
    if srv.is_running:
        srv.stop()


# ===== Graph Fixtures =====

@pytest.fixture
def sample_graph():
    """Create a sample execution graph."""
    from neuroscope.core.graph import ExecutionGraph, GraphNode, GraphEdge

    graph = ExecutionGraph()
    
    node1 = GraphNode(id="input", label="input", module_type="Input")
    node2 = GraphNode(id="linear1", label="fc1", module_type="Linear", parent_id="input")
    node3 = GraphNode(id="relu", label="relu", module_type="ReLU")
    node4 = GraphNode(id="linear2", label="fc2", module_type="Linear")
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    
    graph.add_edge(GraphEdge(source_id="input", target_id="linear1"))
    graph.add_edge(GraphEdge(source_id="linear1", target_id="relu"))
    graph.add_edge(GraphEdge(source_id="relu", target_id="linear2"))
    
    return graph


# ===== Pytest Configuration =====

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
