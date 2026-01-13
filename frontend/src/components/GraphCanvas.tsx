import { useCallback, useEffect, useMemo, useRef } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    useReactFlow,
    type NodeTypes,
} from '@xyflow/react';
import dagre from 'dagre';
import { CustomNode } from './CustomNode';
import { useGraphStore, type GraphNodeData } from '../store/graphStore';
import type { Node, Edge } from '@xyflow/react';

const nodeTypes: NodeTypes = {
    custom: CustomNode,
};

const NODE_WIDTH = 220;
const NODE_HEIGHT = 80;

function getLayoutedElements(
    nodes: Node<GraphNodeData>[],
    edges: Edge[],
    direction = 'TB'
): { nodes: Node<GraphNodeData>[]; edges: Edge[] } {
    if (nodes.length === 0) {
        return { nodes: [], edges: [] };
    }

    // Create a new graph for each layout
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: direction, nodesep: 50, ranksep: 80 });

    // Add nodes
    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
    });

    // Add edges
    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });

    // Run layout
    dagre.layout(dagreGraph);

    // Apply positions
    const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        if (!nodeWithPosition) {
            return { ...node, position: { x: 0, y: 0 } };
        }
        return {
            ...node,
            position: {
                x: nodeWithPosition.x - NODE_WIDTH / 2,
                y: nodeWithPosition.y - NODE_HEIGHT / 2,
            },
        };
    });

    return { nodes: layoutedNodes, edges };
}

export function GraphCanvas() {
    const { nodes: storeNodes, edges: storeEdges, selectNode, selectedNodeId } = useGraphStore();
    const [nodes, setNodes, onNodesChange] = useNodesState<GraphNodeData>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const { fitView } = useReactFlow();

    // Track previous node count to detect actual changes
    const prevNodeCountRef = useRef(0);

    // Apply layout when store changes - use stable effect
    useEffect(() => {
        // Only update if actual change in node count
        if (storeNodes.length === prevNodeCountRef.current && storeNodes.length > 0) {
            return;
        }
        prevNodeCountRef.current = storeNodes.length;

        if (storeNodes.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        console.log('[GraphCanvas] Applying layout to', storeNodes.length, 'nodes');

        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
            storeNodes,
            storeEdges,
            'TB'
        );

        setNodes(layoutedNodes);
        setEdges(layoutedEdges);

        // Fit view after layout with a delay
        const timer = setTimeout(() => {
            fitView({ padding: 0.2 });
        }, 100);

        return () => clearTimeout(timer);
    }, [storeNodes, storeEdges]); // Remove setNodes, setEdges, fitView from deps

    const onNodeClick = useCallback(
        (_: React.MouseEvent, node: Node<GraphNodeData>) => {
            selectNode(node.id);
        },
        [selectNode]
    );

    const onPaneClick = useCallback(() => {
        selectNode(null);
    }, [selectNode]);

    // Highlight selected node
    const nodesWithSelection = useMemo(
        () =>
            nodes.map((node) => ({
                ...node,
                selected: node.id === selectedNodeId,
            })),
        [nodes, selectedNodeId]
    );

    return (
        <div className="graph-canvas">
            <ReactFlow
                nodes={nodesWithSelection}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={onNodeClick}
                onPaneClick={onPaneClick}
                nodeTypes={nodeTypes}
                fitView
                minZoom={0.1}
                maxZoom={2}
                defaultEdgeOptions={{
                    type: 'smoothstep',
                    animated: false,
                }}
            >
                <Background color="var(--bg-dots)" gap={20} size={1} />
                <Controls showInteractive={false} />
                <MiniMap
                    nodeColor={(node) => {
                        const data = node.data as GraphNodeData;
                        if (data?.has_error) return 'var(--node-error)';
                        return `var(--node-${data?.node_type || 'other'})`;
                    }}
                    maskColor="rgba(0, 0, 0, 0.7)"
                />
            </ReactFlow>

            {nodes.length === 0 && (
                <div className="empty-state">
                    <div className="empty-state__icon">N</div>
                    <h2>Waiting for model...</h2>
                    <p>Run: python examples/simple_mlp.py</p>
                </div>
            )}
        </div>
    );
}
