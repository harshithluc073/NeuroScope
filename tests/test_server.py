"""Tests for the WebSocket server."""

import pytest
import asyncio
import json
import time


class TestNeuroScopeServer:
    """Test suite for NeuroScopeServer."""

    @pytest.fixture
    def server(self):
        """Create a server instance."""
        from neuroscope.core.server import NeuroScopeServer

        srv = NeuroScopeServer(port=9999)
        yield srv
        # Cleanup after test
        if srv.is_running:
            srv.stop()

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.host == "localhost"
        assert server.port == 9999
        assert not server.is_running
        assert server.client_count == 0

    def test_url_property(self, server):
        """Test URL generation."""
        assert server.url == "ws://localhost:9999"

    def test_start_stop(self, server):
        """Test starting and stopping the server."""
        # Start server
        server.start(open_browser=False)
        assert server.is_running

        # Stop server
        server.stop()
        assert not server.is_running

    def test_double_start(self, server):
        """Test that double start is handled gracefully."""
        server.start(open_browser=False)
        server.start(open_browser=False)  # Should not raise
        assert server.is_running

    def test_broadcast_when_stopped(self, server):
        """Test that broadcast is safe when server is stopped."""
        # Should not raise
        server.broadcast("test", {"key": "value"})

    @pytest.mark.asyncio
    async def test_websocket_connection(self, server):
        """Test WebSocket client connection."""
        import websockets

        server.start(open_browser=False)
        await asyncio.sleep(0.5)  # Give server time to start

        try:
            async with websockets.connect(server.url) as ws:
                # Should receive welcome message
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(message)

                assert data["type"] == "connected"
                assert "data" in data
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")

    @pytest.mark.asyncio
    async def test_broadcast_message(self, server):
        """Test broadcasting messages to clients."""
        import websockets

        server.start(open_browser=False)
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect(server.url) as ws:
                # Receive welcome message
                await asyncio.wait_for(ws.recv(), timeout=2.0)

                # Broadcast a message
                await asyncio.sleep(0.1)  # Ensure client is registered
                server.broadcast("test_event", {"value": 42})

                # Receive broadcast
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(message)

                assert data["type"] == "test_event"
                assert data["data"]["value"] == 42
        except Exception as e:
            pytest.fail(f"Broadcast test failed: {e}")

    @pytest.mark.asyncio
    async def test_ping_pong(self, server):
        """Test ping/pong handling."""
        import websockets

        server.start(open_browser=False)
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect(server.url) as ws:
                # Skip welcome message
                await asyncio.wait_for(ws.recv(), timeout=2.0)

                # Send ping
                await ws.send(json.dumps({"type": "ping"}))

                # Receive pong
                message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(message)

                assert data["type"] == "pong"
        except Exception as e:
            pytest.fail(f"Ping/pong test failed: {e}")

    @pytest.mark.asyncio
    async def test_client_count(self, server):
        """Test client counting."""
        import websockets

        server.start(open_browser=False)
        await asyncio.sleep(0.5)

        assert server.client_count == 0

        async with websockets.connect(server.url) as ws:
            # Wait for welcome message
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            await asyncio.sleep(0.2)  # Allow registration
            
            # Client should be counted
            assert server.client_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_clients(self, server):
        """Test handling multiple simultaneous clients."""
        import websockets

        server.start(open_browser=False)
        await asyncio.sleep(0.5)

        async with websockets.connect(server.url) as ws1:
            await asyncio.wait_for(ws1.recv(), timeout=2.0)
            
            async with websockets.connect(server.url) as ws2:
                await asyncio.wait_for(ws2.recv(), timeout=2.0)
                await asyncio.sleep(0.2)

                # Both clients should receive broadcast
                server.broadcast("multi_test", {"data": "hello"})

                msg1 = await asyncio.wait_for(ws1.recv(), timeout=2.0)
                msg2 = await asyncio.wait_for(ws2.recv(), timeout=2.0)

                assert json.loads(msg1)["type"] == "multi_test"
                assert json.loads(msg2)["type"] == "multi_test"
