import { create } from 'zustand';
import type { Node, Edge } from '@xyflow/react';

// Types matching Python backend
export interface TensorMetadata {
    shape: number[];
    dtype: string;
    device: string;
    requires_grad: boolean;
    memory_bytes: number;
    is_contiguous: boolean;
}

// v0.2.0: Tensor statistics
export interface TensorStats {
    min_val: number;
    max_val: number;
    mean_val: number;
    std_val: number;
    num_zeros: number;
    num_nan: number;
    num_inf: number;
}

export interface GraphNodeData {
    id: string;
    label: string;
    module_type: string;
    node_type: string;
    depth: number;
    parent_id: string | null;
    children_ids: string[];
    input_tensors: TensorMetadata[];
    output_tensors: TensorMetadata[];
    gradient_tensors: TensorMetadata[];  // v0.2.0
    execution_order: number;
    execution_time_ms: number;  // v0.2.0
    memory_delta_bytes: number;  // v0.2.0
    has_error: boolean;
    error_message: string | null;
    tensor_stats: TensorStats | null;  // v0.2.0
    extra_info: Record<string, unknown>;
    // Index signature for React Flow Node compatibility
    [key: string]: unknown;
}

export interface ExecutionGraph {
    nodes: Record<string, GraphNodeData>;
    edges: Array<{
        source_id: string;
        target_id: string;
        source_output_idx: number;
        target_input_idx: number;
        tensor_info: TensorMetadata | null;
    }>;
    metadata: Record<string, unknown>;
    timestamp: number;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// v0.2.0: View modes for different visualization types
export type ViewMode = 'normal' | 'profiling' | 'gradients';

interface GraphStore {
    // Graph data (React Flow format)
    nodes: Node<GraphNodeData>[];
    edges: Edge[];

    // Original data from backend
    graphData: ExecutionGraph | null;

    // UI State
    selectedNodeId: string | null;
    collapsedGroups: Set<string>;
    connectionStatus: ConnectionStatus;
    searchQuery: string;
    viewMode: ViewMode;  // v0.2.0

    // Actions
    setGraph: (graph: ExecutionGraph) => void;
    addNode: (node: GraphNodeData) => void;
    addNodes: (nodes: GraphNodeData[]) => void;  // Batch add for performance
    updateNodeGradients: (nodeId: string, gradients: TensorMetadata[]) => void;  // v0.2.0
    selectNode: (nodeId: string | null) => void;
    toggleCollapse: (groupId: string) => void;
    setConnectionStatus: (status: ConnectionStatus) => void;
    setSearchQuery: (query: string) => void;
    setViewMode: (mode: ViewMode) => void;  // v0.2.0
    clearGraph: () => void;
}

// Convert backend graph to React Flow format
function convertToReactFlow(graph: ExecutionGraph): { nodes: Node<GraphNodeData>[]; edges: Edge[] } {
    const nodes: Node<GraphNodeData>[] = Object.values(graph.nodes).map((node, index) => ({
        id: node.id,
        type: 'custom',
        position: { x: 0, y: 0 }, // Will be calculated by layout
        data: node,
    }));

    const edges: Edge[] = graph.edges.map((edge, index) => ({
        id: `e-${edge.source_id}-${edge.target_id}-${index}`,
        source: edge.source_id,
        target: edge.target_id,
        animated: false,
        style: { stroke: 'var(--edge-default)' },
    }));

    return { nodes, edges };
}

export const useGraphStore = create<GraphStore>((set, get) => ({
    nodes: [],
    edges: [],
    graphData: null,
    selectedNodeId: null,
    collapsedGroups: new Set(),
    connectionStatus: 'disconnected',
    searchQuery: '',
    viewMode: 'normal',  // v0.2.0

    setGraph: (graph) => {
        const { nodes, edges } = convertToReactFlow(graph);
        set({
            graphData: graph,
            nodes,
            edges,
        });
    },

    addNode: (nodeData) => {
        set((state) => {
            // Add or update node
            const existingIndex = state.nodes.findIndex((n) => n.id === nodeData.id);
            const newNode: Node<GraphNodeData> = {
                id: nodeData.id,
                type: 'custom',
                position: { x: 0, y: 0 },
                data: nodeData,
            };

            let newNodes: Node<GraphNodeData>[];
            if (existingIndex >= 0) {
                newNodes = [...state.nodes];
                newNodes[existingIndex] = newNode;
            } else {
                newNodes = [...state.nodes, newNode];
            }

            return { nodes: newNodes };
        });
    },

    // Efficient batch add - single state update for multiple nodes
    addNodes: (nodesData) => {
        set((state) => {
            const nodeMap = new Map(state.nodes.map(n => [n.id, n]));

            for (const nodeData of nodesData) {
                const newNode: Node<GraphNodeData> = {
                    id: nodeData.id,
                    type: 'custom',
                    position: { x: 0, y: 0 },
                    data: nodeData,
                };
                nodeMap.set(nodeData.id, newNode);
            }

            return { nodes: Array.from(nodeMap.values()) };
        });
    },


    selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

    toggleCollapse: (groupId) => {
        set((state) => {
            const newCollapsed = new Set(state.collapsedGroups);
            if (newCollapsed.has(groupId)) {
                newCollapsed.delete(groupId);
            } else {
                newCollapsed.add(groupId);
            }
            return { collapsedGroups: newCollapsed };
        });
    },

    setConnectionStatus: (status) => set({ connectionStatus: status }),

    setSearchQuery: (query) => set({ searchQuery: query }),

    // v0.2.0: Set view mode (normal, profiling, gradients)
    setViewMode: (mode) => set({ viewMode: mode }),

    // v0.2.0: Update node gradients from backward pass
    updateNodeGradients: (nodeId, gradients) => {
        set((state) => {
            const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
            if (nodeIndex < 0) return state;

            const newNodes = [...state.nodes];
            const node = newNodes[nodeIndex];
            newNodes[nodeIndex] = {
                ...node,
                data: {
                    ...node.data,
                    gradient_tensors: gradients,
                },
            };
            return { nodes: newNodes };
        });
    },

    clearGraph: () => set({ nodes: [], edges: [], graphData: null, selectedNodeId: null, viewMode: 'normal' }),
}));
