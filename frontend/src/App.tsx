import { useEffect } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { GraphCanvas } from './components/GraphCanvas';
import { NodeInspector } from './components/NodeInspector';
import { Header } from './components/Header';
import { ConnectionStatus } from './components/ConnectionStatus';
import { ErrorBoundary } from './components/ErrorBoundary';
import { useGraphStore, type ExecutionGraph, type GraphNodeData } from './store/graphStore';
import { useWebSocket } from './hooks/useWebSocket';

import '@xyflow/react/dist/style.css';

export function App() {
    const { setGraph, addNode, addNodes, setConnectionStatus } = useGraphStore();

    const { status, connect } = useWebSocket({
        url: 'ws://localhost:8765',
        onMessage: (message) => {
            try {
                if (message.type === 'graph_update') {
                    const graphData = message.data as ExecutionGraph;
                    if (graphData && graphData.nodes) {
                        setGraph(graphData);
                    }
                } else if (message.type === 'node_update') {
                    const nodeData = message.data as GraphNodeData;
                    if (nodeData && nodeData.id) {
                        addNode(nodeData);
                    }
                } else if (message.type === 'batch_node_update') {
                    // Handle batched node updates for performance
                    const batch = message.data as { nodes: GraphNodeData[], count: number };
                    if (batch && batch.nodes && batch.nodes.length > 0) {
                        addNodes(batch.nodes);
                    }
                }
            } catch (error) {
                console.error('[NeuroScope] Error processing message:', error);
            }
        },
        onStatusChange: setConnectionStatus,
    });

    useEffect(() => {
        connect();
    }, [connect]);

    return (
        <ErrorBoundary>
            <div className="app">
                <Header />
                <ConnectionStatus status={status} />
                <div className="main-content">
                    <ErrorBoundary>
                        <ReactFlowProvider>
                            <GraphCanvas />
                        </ReactFlowProvider>
                    </ErrorBoundary>
                    <NodeInspector />
                </div>
            </div>
        </ErrorBoundary>
    );
}
