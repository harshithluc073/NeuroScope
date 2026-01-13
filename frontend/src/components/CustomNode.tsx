import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '../store/graphStore';

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

export const CustomNode = memo(function CustomNode({
    data,
    selected,
}: NodeProps<GraphNodeData>) {
    const icon = NODE_ICONS[data.node_type] || NODE_ICONS.other;
    const inputShape = data.input_tensors[0]?.shape;
    const outputShape = data.output_tensors[0]?.shape;
    const device = data.output_tensors[0]?.device || '';
    const memory = data.output_tensors[0]?.memory_bytes || 0;

    return (
        <div
            className={`custom-node custom-node--${data.node_type} ${data.has_error ? 'custom-node--error' : ''
                } ${selected ? 'custom-node--selected' : ''}`}
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
            </div>

            {data.has_error && data.error_message && (
                <div className="custom-node__error-message">{data.error_message}</div>
            )}

            <Handle type="source" position={Position.Bottom} />
        </div>
    );
});
