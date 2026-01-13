"""
WebSocket server for real-time graph streaming.

Runs in a background thread to avoid blocking the training loop.
Broadcasts graph updates to all connected browser clients.

Security features:
- Optional API key authentication
- CORS origin validation
- Rate limiting
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

# Reduce websockets library logging noise
logging.getLogger('websockets').setLevel(logging.WARNING)

# Simple logger for NeuroScope
logger = logging.getLogger('neuroscope.server')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[NeuroScope] %(message)s'))
    logger.addHandler(handler)


class NeuroScopeServer:
    """
    WebSocket server for streaming graph updates to browser clients.

    The server runs in a background thread and broadcasts messages
    to all connected clients.
    
    Security features:
    - api_key: If set, clients must provide this key to connect
    - allowed_origins: List of allowed CORS origins (None = allow all)
    - rate_limit: Max messages per second per client
    """

    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 8765,
        api_key: str | None = None,
        allowed_origins: list[str] | None = None,
        rate_limit: int = 100,
    ) -> None:
        """
        Initialize the NeuroScope server.
        
        Args:
            host: Host to bind to (default: localhost)
            port: Port to bind to (default: 8765)
            api_key: Optional API key for authentication. Set via NEUROSCOPE_API_KEY env var
                    or pass directly. If None, no authentication required.
            allowed_origins: List of allowed CORS origins. None = allow all.
            rate_limit: Max messages per second per client (default: 100)
        """
        self.host = host
        self.port = port
        
        # Security settings
        self._api_key = api_key or os.environ.get("NEUROSCOPE_API_KEY")
        self._allowed_origins = allowed_origins
        self._rate_limit = rate_limit
        
        # Rate limiting state
        self._client_message_counts: dict[str, list[float]] = {}

        self._clients: set[WebSocketServerProtocol] = set()
        self._authenticated_clients: set[str] = set()
        self._server: websockets.WebSocketServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def requires_auth(self) -> bool:
        return self._api_key is not None

    def generate_api_key(self) -> str:
        """Generate a new random API key."""
        return secrets.token_urlsafe(32)

    def start(self, open_browser: bool = True) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for server to start
        time.sleep(0.3)

        if self.requires_auth:
            logger.info(f"Server started with API key authentication")
        
        if open_browser:
            self._open_browser()

    def stop(self) -> None:
        """Stop the server and disconnect all clients."""
        if not self._running:
            return

        self._running = False

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._clients.clear()
        self._authenticated_clients.clear()
        self._client_message_counts.clear()

    def broadcast(self, message_type: str, data: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self._running or self._loop is None:
            return

        message = {
            "type": message_type,
            "timestamp": time.time(),
            "data": data,
        }

        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(message),
            self._loop
        )

    async def _broadcast_async(self, message: dict[str, Any]) -> None:
        """Internal async broadcast implementation."""
        if not self._clients:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for client in self._clients:
            try:
                await client.send(message_json)
            except websockets.ConnectionClosed:
                disconnected.add(client)
            except Exception:
                disconnected.add(client)

        self._clients -= disconnected

    def _run_server(self) -> None:
        """Run the server event loop in a thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _serve(self) -> None:
        """Start the WebSocket server."""
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            process_request=self._process_request if self._allowed_origins else None,
        ) as server:
            self._server = server

            while self._running:
                await asyncio.sleep(0.1)

    async def _process_request(self, path: str, headers: Any) -> tuple | None:
        """Process HTTP request for CORS validation."""
        if self._allowed_origins is None:
            return None
            
        origin = headers.get("Origin", "")
        if origin and origin not in self._allowed_origins:
            logger.warning(f"Rejected connection from origin: {origin}")
            return (403, [("Content-Type", "text/plain")], b"Forbidden: Invalid origin")
        
        return None

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        is_authenticated = not self.requires_auth

        try:
            # If auth required, wait for auth message first
            if self.requires_auth:
                try:
                    auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    auth_data = json.loads(auth_message)
                    
                    if auth_data.get("type") == "auth" and auth_data.get("api_key") == self._api_key:
                        is_authenticated = True
                        self._authenticated_clients.add(client_id)
                        await websocket.send(json.dumps({
                            "type": "auth_success",
                            "timestamp": time.time(),
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "auth_failed",
                            "timestamp": time.time(),
                            "data": {"error": "Invalid API key"},
                        }))
                        return
                except asyncio.TimeoutError:
                    await websocket.send(json.dumps({
                        "type": "auth_failed",
                        "timestamp": time.time(),
                        "data": {"error": "Authentication timeout"},
                    }))
                    return

            self._clients.add(websocket)
            self._client_message_counts[client_id] = []

            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connected",
                "timestamp": time.time(),
                "data": {"message": "Connected to NeuroScope server"},
            }))

            # Handle incoming messages
            async for message in websocket:
                if not self._check_rate_limit(client_id):
                    await websocket.send(json.dumps({
                        "type": "error",
                        "timestamp": time.time(),
                        "data": {"error": "Rate limit exceeded"},
                    }))
                    continue
                    
                await self._handle_message(websocket, message)

        except websockets.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            self._authenticated_clients.discard(client_id)
            self._client_message_counts.pop(client_id, None)

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_start = now - 1.0  # 1 second window
        
        # Get message timestamps for this client
        timestamps = self._client_message_counts.get(client_id, [])
        
        # Remove old timestamps
        timestamps = [t for t in timestamps if t > window_start]
        
        # Check limit
        if len(timestamps) >= self._rate_limit:
            return False
        
        # Add new timestamp
        timestamps.append(now)
        self._client_message_counts[client_id] = timestamps
        
        return True

    async def _handle_message(
        self, websocket: WebSocketServerProtocol, message: str
    ) -> None:
        """Handle an incoming message from a client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")

            if msg_type == "ping":
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": time.time(),
                }))
        except json.JSONDecodeError:
            pass

    def _open_browser(self) -> None:
        """Open the frontend in the default browser."""
        frontend_path = Path(__file__).parent.parent.parent / "frontend" / "dist" / "index.html"

        if frontend_path.exists():
            webbrowser.open(f"file://{frontend_path}")
