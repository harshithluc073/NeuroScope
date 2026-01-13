import { useCallback, useRef, useState, useEffect } from 'react';
import type { ConnectionStatus } from '../store/graphStore';

interface WebSocketMessage {
    type: string;
    timestamp: number;
    data: unknown;
}

interface UseWebSocketOptions {
    url: string;
    onMessage: (message: WebSocketMessage) => void;
    onStatusChange: (status: ConnectionStatus) => void;
    reconnectInterval?: number;
}

export function useWebSocket({
    url,
    onMessage,
    onStatusChange,
    reconnectInterval = 5000,
}: UseWebSocketOptions) {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const [status, setStatus] = useState<ConnectionStatus>('disconnected');
    const mountedRef = useRef(true);

    // Store callbacks in refs to avoid dependency issues
    const onMessageRef = useRef(onMessage);
    const onStatusChangeRef = useRef(onStatusChange);

    useEffect(() => {
        onMessageRef.current = onMessage;
        onStatusChangeRef.current = onStatusChange;
    }, [onMessage, onStatusChange]);

    const updateStatus = useCallback((newStatus: ConnectionStatus) => {
        if (!mountedRef.current) return;
        setStatus(newStatus);
        onStatusChangeRef.current(newStatus);
    }, []);

    const connect = useCallback(() => {
        // Prevent multiple connections
        if (wsRef.current?.readyState === WebSocket.OPEN ||
            wsRef.current?.readyState === WebSocket.CONNECTING) {
            return;
        }

        updateStatus('connecting');

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                if (!mountedRef.current) return;
                console.log('[NeuroScope] Connected to server');
                updateStatus('connected');

                if (reconnectTimeoutRef.current) {
                    clearTimeout(reconnectTimeoutRef.current);
                    reconnectTimeoutRef.current = null;
                }
            };

            ws.onmessage = (event) => {
                if (!mountedRef.current) return;
                try {
                    const message = JSON.parse(event.data) as WebSocketMessage;
                    console.log('[NeuroScope] Received:', message.type);
                    onMessageRef.current(message);
                } catch (e) {
                    console.error('[NeuroScope] Failed to parse message:', e);
                }
            };

            ws.onerror = () => {
                if (!mountedRef.current) return;
                console.log('[NeuroScope] WebSocket error - server may not be running');
                updateStatus('error');
            };

            ws.onclose = () => {
                if (!mountedRef.current) return;
                console.log('[NeuroScope] Disconnected');
                updateStatus('disconnected');
                wsRef.current = null;

                // Attempt to reconnect after delay
                if (mountedRef.current) {
                    reconnectTimeoutRef.current = window.setTimeout(() => {
                        if (mountedRef.current) {
                            connect();
                        }
                    }, reconnectInterval);
                }
            };
        } catch (e) {
            console.error('[NeuroScope] Failed to connect:', e);
            updateStatus('error');
        }
    }, [url, updateStatus, reconnectInterval]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        updateStatus('disconnected');
    }, [updateStatus]);

    // Cleanup on unmount
    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    const send = useCallback((message: unknown) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        }
    }, []);

    return {
        status,
        connect,
        disconnect,
        send,
    };
}
