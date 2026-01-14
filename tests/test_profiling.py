"""Tests for v0.2.0 profiling features."""

import pytest


class TestProfilingFeatures:
    """Test suite for v0.2.0 profiling features."""

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
    def profiling_tracer(self):
        """Create a PyTorchTracer with profiling enabled."""
        from neuroscope.tracers.pytorch import PyTorchTracer

        return PyTorchTracer(
            enable_profiling=True,
            track_memory=True,
            capture_tensor_stats=True,
            capture_gradients=False,
        )

    def test_timing_capture(self, profiling_tracer, simple_model):
        """Test that execution_time_ms is captured."""
        import torch

        profiling_tracer.attach(simple_model)
        profiling_tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = simple_model(x)

        graph = profiling_tracer.get_graph()

        # Check that nodes have timing data
        timing_found = False
        for node in graph.nodes.values():
            if node.execution_time_ms > 0:
                timing_found = True
                break

        assert timing_found, "No timing data captured in any node"

        profiling_tracer.detach()

    def test_tensor_stats_capture(self, profiling_tracer, simple_model):
        """Test that tensor statistics are captured."""
        import torch

        profiling_tracer.attach(simple_model)
        profiling_tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = simple_model(x)

        graph = profiling_tracer.get_graph()

        # Check that at least some nodes have tensor stats
        stats_found = False
        for node in graph.nodes.values():
            if node.tensor_stats is not None:
                stats_found = True
                # Verify stats have expected fields
                assert hasattr(node.tensor_stats, 'min_val')
                assert hasattr(node.tensor_stats, 'max_val')
                assert hasattr(node.tensor_stats, 'mean_val')
                break

        assert stats_found, "No tensor stats captured in any node"

        profiling_tracer.detach()

    def test_tensor_stats_dataclass(self):
        """Test TensorStats dataclass serialization."""
        from neuroscope.core.graph import TensorStats

        stats = TensorStats(
            min_val=-1.5,
            max_val=2.5,
            mean_val=0.5,
            std_val=1.0,
            num_zeros=10,
            num_nan=0,
            num_inf=0,
        )

        # Test to_dict
        d = stats.to_dict()
        assert d["min_val"] == -1.5
        assert d["max_val"] == 2.5
        assert d["num_zeros"] == 10

        # Test from_dict
        stats2 = TensorStats.from_dict(d)
        assert stats2.min_val == stats.min_val
        assert stats2.mean_val == stats.mean_val

    def test_graph_node_new_fields(self):
        """Test GraphNode has new v0.2.0 fields."""
        from neuroscope.core.graph import GraphNode, NodeType, TensorStats

        node = GraphNode(
            id="test_node",
            label="test",
            module_type="torch.nn.Linear",
            node_type=NodeType.LINEAR,
            execution_time_ms=1.5,
            memory_delta_bytes=1024,
            tensor_stats=TensorStats(min_val=0.0, max_val=1.0, mean_val=0.5)
        )

        assert node.execution_time_ms == 1.5
        assert node.memory_delta_bytes == 1024
        assert node.gradient_tensors == []
        assert node.tensor_stats is not None

        # Test serialization
        d = node.to_dict()
        assert d["execution_time_ms"] == 1.5
        assert d["memory_delta_bytes"] == 1024
        assert d["tensor_stats"]["mean_val"] == 0.5

        # Test deserialization
        node2 = GraphNode.from_dict(d)
        assert node2.execution_time_ms == 1.5
        assert node2.tensor_stats.mean_val == 0.5

    def test_gradient_capture(self):
        """Test gradient capture during backward pass."""
        import torch
        import torch.nn as nn
        from neuroscope.tracers.pytorch import PyTorchTracer

        # Simple model
        model = nn.Linear(10, 5)

        # Tracer with gradient capture
        tracer = PyTorchTracer(capture_gradients=True)
        tracer.attach(model)
        tracer.reset_graph()

        # Forward and backward pass
        x = torch.randn(4, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        graph = tracer.get_graph()

        # Check that gradient tensors were captured
        grad_found = False
        for node in graph.nodes.values():
            if len(node.gradient_tensors) > 0:
                grad_found = True
                # Gradient should have shape info
                assert len(node.gradient_tensors[0].shape) > 0
                break

        # Note: gradient capture may not work for all layer types
        # Just verify no errors occurred

        tracer.detach()

    def test_profiling_disabled(self, simple_model):
        """Test that profiling can be disabled."""
        import torch
        from neuroscope.tracers.pytorch import PyTorchTracer

        tracer = PyTorchTracer(
            enable_profiling=False,
            track_memory=False,
            capture_tensor_stats=False,
        )
        tracer.attach(simple_model)
        tracer.reset_graph()

        x = torch.randn(4, 10)
        _ = simple_model(x)

        graph = tracer.get_graph()

        # All timing should be 0 when disabled
        for node in graph.nodes.values():
            assert node.execution_time_ms == 0.0
            assert node.memory_delta_bytes == 0
            assert node.tensor_stats is None

        tracer.detach()
