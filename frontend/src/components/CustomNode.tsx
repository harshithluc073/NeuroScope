import { memo, useMemo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useGraphStore, type GraphNodeData } from '../store/graphStore';

// Node type icons
const NODE_ICONS: Record<string, string> = {
    convolution: '🔲',
    linear: '➡️',
    activation: '⚡',
    normalization: '📊',
    pooling: '🔽',
    attention: '👁️',
    embedding: '📝',
    dropout: '💧',
    recurrent: '🔄',
    container: '📦',
    other: '⚙️',
};

function formatShape(shape: number[]): string {
    return `[${shape.join(', ')}]`;
}

function formatMemory(bytes: number): string {
    if (bytes === 0) return '';
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function formatTime(ms: number): string {
    if (ms === 0) return '';
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

// Get heatmap color based on execution time (0-1 normalized)
function getHeatmapColor(normalizedValue: number): string {
    // Blue (cold) -> Yellow -> Red (hot)
    const hue = (1 - normalizedValue) * 240; // 240=blue, 0=red
    return `hsl(${hue}, 80%, 50%)`;
}

export const CustomNode = memo(function CustomNode({
    data,
    selected,
}: NodeProps<GraphNodeData>) {
    const viewMode = useGraphStore((state) => state.viewMode);
    const nodes = useGraphStore((state) => state.nodes);

    // Calculate max execution time for normalization (only in profiling mode)
    const maxExecutionTime = useMemo(() => {
        if (viewMode !== 'profiling') return 0;
        return Math.max(...nodes.map((n) => (n.data as GraphNodeData).execution_time_ms || 0), 0.001);
    }, [viewMode, nodes]);

    const icon = NODE_ICONS[data.node_type] || NODE_ICONS.other;
    const inputShape = data.input_tensors?.[0]?.shape;
    const outputShape = data.output_tensors?.[0]?.shape;
    const device = data.output_tensors?.[0]?.device || '';
    const memory = data.output_tensors?.[0]?.memory_bytes || 0;

    // v0.2.0: Profiling data
    const executionTime = data.execution_time_ms || 0;
    const memoryDelta = data.memory_delta_bytes || 0;
    const hasGradients = (data.gradient_tensors?.length || 0) > 0;

    // Dynamic styles based on view mode
    const profilingStyle = useMemo(() => {
        if (viewMode !== 'profiling' || executionTime === 0) return {};
        const normalizedTime = executionTime / maxExecutionTime;
        return {
            borderColor: getHeatmapColor(normalizedTime),
            borderWidth: '3px',
            boxShadow: `0 0 ${8 + normalizedTime * 12}px ${getHeatmapColor(normalizedTime)}40`,
        };
    }, [viewMode, executionTime, maxExecutionTime]);

    const gradientStyle = useMemo(() => {
        if (viewMode !== 'gradients') return {};
        return hasGradients
            ? { borderColor: '#22c55e', borderWidth: '3px' }
            : { opacity: 0.5 };
    }, [viewMode, hasGradients]);

    return (
        <div
            className={`custom-node custom-node--${data.node_type} ${data.has_error ? 'custom-node--error' : ''
                } ${selected ? 'custom-node--selected' : ''}`}
            style={{ ...profilingStyle, ...gradientStyle }}
        >
            <Handle type="target" position={Position.Top} />

            <div className="custom-node__header">
                <span className="custom-node__icon">{icon}</span>
                <span className="custom-node__label" title={data.label}>
                    {data.label || data.module_type.split('.').pop()}
                </span>
                {data.has_error && <span className="custom-node__error-badge">!</span>}
            </div>

            <div className="custom-node__body">
                <div className="custom-node__type">{data.module_type.split('.').pop()}</div>

                <div className="custom-node__tensors">
                    {inputShape && (
                        <span className="custom-node__tensor custom-node__tensor--in">
                            {formatShape(inputShape)}
                        </span>
                    )}
                    {inputShape && outputShape && <span className="custom-node__arrow">→</span>}
                    {outputShape && (
                        <span className="custom-node__tensor custom-node__tensor--out">
                            {formatShape(outputShape)}
                        </span>
                    )}
                </div>

                <div className="custom-node__meta">
                    {device && <span className="custom-node__device">{device}</span>}
                    {memory > 0 && <span className="custom-node__memory">{formatMemory(memory)}</span>}
                </div>

                {/* v0.2.0: Profiling info */}
                {viewMode === 'profiling' && executionTime > 0 && (
                    <div className="custom-node__profiling">
                        <span className="custom-node__time" title="Execution time">
                            ⏱️ {formatTime(executionTime)}
                        </span>
                        {memoryDelta !== 0 && (
                            <span className={`custom-node__memory-delta ${memoryDelta > 0 ? 'positive' : 'negative'}`} title="Memory delta">
                                {memoryDelta > 0 ? '+' : ''}{formatMemory(Math.abs(memoryDelta))}
                            </span>
                        )}
                    </div>
                )}

                {/* v0.2.0: Gradient indicator */}
                {viewMode === 'gradients' && hasGradients && (
                    <div className="custom-node__gradient-indicator">
                        <span className="custom-node__gradient-badge">∇</span>
                    </div>
                )}
            </div>

            {data.has_error && data.error_message && (
                <div className="custom-node__error-message">{data.error_message}</div>
            )}

            <Handle type="source" position={Position.Bottom} />
        </div>
    );
});
