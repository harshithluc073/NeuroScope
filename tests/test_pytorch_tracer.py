"""Tests for the PyTorch tracer."""

import pytest


class TestPyTorchTracer:
    """Test suite for PyTorchTracer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    @pytest.fixture
    def tracer(self):
        """Create a PyTorchTracer instance."""
        from neuroscope.tracers.pytorch import PyTorchTracer

        return PyTorchTracer()

    def test_attach_detach(self, tracer, simple_model):
        """Test attaching and detaching from a model."""
        assert not tracer.is_attached

        tracer.attach(simple_model)
        assert tracer.is_attached

        tracer.detach()
        assert not tracer.is_attached

    def test_double_attach_raises(self, tracer, simple_model):
        """Test that attaching twice raises an error."""
        tracer.attach(simple_model)

        with pytest.raises(RuntimeError, match="Already attached"):
            tracer.attach(simple_model)

        tracer.detach()

    def test_attach_invalid_type(self, tracer):
        """Test that attaching to non-Module raises ValueError."""
        with pytest.raises(ValueError, match="Expected torch.nn.Module"):
            tracer.attach("not a model")

    def test_forward_capture(self, tracer, simple_model):
        """Test that forward pass captures nodes."""
        import torch

        tracer.attach(simple_model)
        tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = simple_model(x)

        graph = tracer.get_graph()

        # Should have nodes for each layer
        assert len(graph.nodes) > 0

        tracer.detach()

    def test_tensor_metadata(self, tracer, simple_model):
        """Test that tensor metadata is captured correctly."""
        import torch

        tracer.attach(simple_model)
        tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = simple_model(x)

        graph = tracer.get_graph()

        # Check that nodes have tensor metadata
        for node in graph.nodes.values():
            if node.output_tensors:
                tensor_meta = node.output_tensors[0]
                assert len(tensor_meta.shape) > 0
                assert tensor_meta.dtype != ""
                assert tensor_meta.device != ""

        tracer.detach()

    def test_nan_detection(self, tracer):
        """Test that NaN values are detected."""
        import torch
        import torch.nn as nn

        # Create a model that produces NaN
        class NaNModel(nn.Module):
            def forward(self, x):
                return x / 0.0  # Will produce inf/nan

        model = NaNModel()
        tracer.attach(model)
        tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = model(x)

        graph = tracer.get_graph()
        error_nodes = graph.get_error_nodes()

        assert len(error_nodes) > 0

        tracer.detach()

    def test_node_hierarchy(self, tracer):
        """Test that nested modules create proper hierarchy."""
        import torch
        import torch.nn as nn

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                )
                self.fc = nn.Linear(20, 5)

            def forward(self, x):
                x = self.block(x)
                return self.fc(x)

        model = NestedModel()
        tracer.attach(model)
        tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = model(x)

        graph = tracer.get_graph()

        # Check for proper depth values
        depths = [node.depth for node in graph.nodes.values()]
        assert max(depths) > 0  # Should have nested nodes

        tracer.detach()

    def test_module_classification(self, tracer):
        """Test that module types are classified correctly."""
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        tracer.attach(model)
        tracer.reset_graph()

        x = torch.randn(1, 3, 32, 32)
        _ = model(x)

        graph = tracer.get_graph()

        node_types = {
            node.label: node.node_type.value for node in graph.nodes.values()
        }

        # Should have various node types
        assert any("conv" in t for t in node_types.values())

        tracer.detach()
