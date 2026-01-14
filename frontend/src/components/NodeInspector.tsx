import { useGraphStore, type TensorMetadata, type GraphNodeData, type TensorStats } from '../store/graphStore';

function formatShape(shape: number[]): string {
    return `[${shape.join(', ')}]`;
}

function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(ms: number): string {
    if (ms === 0) return '0 ms';
    if (ms < 1) return `${(ms * 1000).toFixed(0)} μs`;
    if (ms < 1000) return `${ms.toFixed(3)} ms`;
    return `${(ms / 1000).toFixed(3)} s`;
}

function TensorInfo({ tensor, label }: { tensor: TensorMetadata; label: string }) {
    return (
        <div className="tensor-info">
            <div className="tensor-info__label">{label}</div>
            <div className="tensor-info__details">
                <div className="tensor-info__row">
                    <span className="tensor-info__key">Shape</span>
                    <code className="tensor-info__value">{formatShape(tensor.shape)}</code>
                </div>
                <div className="tensor-info__row">
                    <span className="tensor-info__key">DType</span>
                    <code className="tensor-info__value">{tensor.dtype}</code>
                </div>
                <div className="tensor-info__row">
                    <span className="tensor-info__key">Device</span>
                    <code className="tensor-info__value">{tensor.device}</code>
                </div>
                {tensor.memory_bytes > 0 && (
                    <div className="tensor-info__row">
                        <span className="tensor-info__key">Memory</span>
                        <code className="tensor-info__value">{formatBytes(tensor.memory_bytes)}</code>
                    </div>
                )}
                <div className="tensor-info__row">
                    <span className="tensor-info__key">Grad</span>
                    <code className="tensor-info__value">{tensor.requires_grad ? 'Yes' : 'No'}</code>
                </div>
            </div>
        </div>
    );
}

// v0.2.0: Tensor Statistics component
function TensorStatsInfo({ stats }: { stats: TensorStats }) {
    return (
        <div className="tensor-stats">
            <div className="tensor-stats__row">
                <span className="tensor-stats__key">Min</span>
                <code className="tensor-stats__value">{stats.min_val.toExponential(3)}</code>
            </div>
            <div className="tensor-stats__row">
                <span className="tensor-stats__key">Max</span>
                <code className="tensor-stats__value">{stats.max_val.toExponential(3)}</code>
            </div>
            <div className="tensor-stats__row">
                <span className="tensor-stats__key">Mean</span>
                <code className="tensor-stats__value">{stats.mean_val.toExponential(3)}</code>
            </div>
            <div className="tensor-stats__row">
                <span className="tensor-stats__key">Std</span>
                <code className="tensor-stats__value">{stats.std_val.toExponential(3)}</code>
            </div>
            {(stats.num_nan > 0 || stats.num_inf > 0 || stats.num_zeros > 0) && (
                <div className="tensor-stats__row tensor-stats__row--warning">
                    <span className="tensor-stats__key">Issues</span>
                    <code className="tensor-stats__value">
                        {stats.num_nan > 0 && `${stats.num_nan} NaN `}
                        {stats.num_inf > 0 && `${stats.num_inf} Inf `}
                        {stats.num_zeros > 0 && `${stats.num_zeros} zeros`}
                    </code>
                </div>
            )}
        </div>
    );
}

export function NodeInspector() {
    const { selectedNodeId, nodes, selectNode } = useGraphStore();

    const selectedNode = nodes.find((n) => n.id === selectedNodeId);

    if (!selectedNode) {
        return (
            <aside className="node-inspector node-inspector--empty">
                <h3>Node Inspector</h3>
                <p className="node-inspector__hint">Click on a node to inspect its details</p>
            </aside>
        );
    }

    const data = selectedNode.data as GraphNodeData;

    return (
        <aside className="node-inspector">
            <div className="node-inspector__header">
                <h3>{data.label}</h3>
                <button className="node-inspector__close" onClick={() => selectNode(null)}>
                    ×
                </button>
            </div>

            <div className="node-inspector__content">
                {/* Basic Info */}
                <section className="node-inspector__section">
                    <h4>Module</h4>
                    <div className="node-inspector__info">
                        <div className="node-inspector__row">
                            <span className="node-inspector__key">Type</span>
                            <code className="node-inspector__value">{data.module_type}</code>
                        </div>
                        <div className="node-inspector__row">
                            <span className="node-inspector__key">Category</span>
                            <span className="node-inspector__value node-inspector__badge" data-type={data.node_type}>
                                {data.node_type}
                            </span>
                        </div>
                        <div className="node-inspector__row">
                            <span className="node-inspector__key">Execution Order</span>
                            <span className="node-inspector__value">{data.execution_order}</span>
                        </div>
                        <div className="node-inspector__row">
                            <span className="node-inspector__key">Depth</span>
                            <span className="node-inspector__value">{data.depth}</span>
                        </div>
                    </div>
                </section>

                {/* v0.2.0: Profiling Info */}
                {(data.execution_time_ms > 0 || data.memory_delta_bytes !== 0) && (
                    <section className="node-inspector__section node-inspector__section--profiling">
                        <h4>⏱️ Profiling</h4>
                        <div className="node-inspector__info">
                            {data.execution_time_ms > 0 && (
                                <div className="node-inspector__row">
                                    <span className="node-inspector__key">Execution Time</span>
                                    <code className="node-inspector__value node-inspector__value--time">
                                        {formatTime(data.execution_time_ms)}
                                    </code>
                                </div>
                            )}
                            {data.memory_delta_bytes !== 0 && (
                                <div className="node-inspector__row">
                                    <span className="node-inspector__key">Memory Delta</span>
                                    <code className={`node-inspector__value ${data.memory_delta_bytes > 0 ? 'positive' : 'negative'}`}>
                                        {data.memory_delta_bytes > 0 ? '+' : ''}{formatBytes(Math.abs(data.memory_delta_bytes))}
                                    </code>
                                </div>
                            )}
                        </div>
                    </section>
                )}

                {/* Error */}
                {data.has_error && (
                    <section className="node-inspector__section node-inspector__section--error">
                        <h4>⚠️ Error Detected</h4>
                        <p className="node-inspector__error">{data.error_message}</p>
                    </section>
                )}

                {/* v0.2.0: Tensor Statistics */}
                {data.tensor_stats && (
                    <section className="node-inspector__section">
                        <h4>📊 Tensor Statistics</h4>
                        <TensorStatsInfo stats={data.tensor_stats} />
                    </section>
                )}

                {/* Input Tensors */}
                {data.input_tensors && data.input_tensors.length > 0 && (
                    <section className="node-inspector__section">
                        <h4>Input Tensors</h4>
                        {data.input_tensors.map((tensor, i) => (
                            <TensorInfo key={i} tensor={tensor} label={`Input ${i}`} />
                        ))}
                    </section>
                )}

                {/* Output Tensors */}
                {data.output_tensors && data.output_tensors.length > 0 && (
                    <section className="node-inspector__section">
                        <h4>Output Tensors</h4>
                        {data.output_tensors.map((tensor, i) => (
                            <TensorInfo key={i} tensor={tensor} label={`Output ${i}`} />
                        ))}
                    </section>
                )}

                {/* v0.2.0: Gradient Tensors */}
                {data.gradient_tensors && data.gradient_tensors.length > 0 && (
                    <section className="node-inspector__section node-inspector__section--gradients">
                        <h4>∇ Gradient Tensors</h4>
                        {data.gradient_tensors.map((tensor, i) => (
                            <TensorInfo key={i} tensor={tensor} label={`Gradient ${i}`} />
                        ))}
                    </section>
                )}

                {/* Extra Info */}
                {data.extra_info && Object.keys(data.extra_info).length > 0 && (
                    <section className="node-inspector__section">
                        <h4>Additional Info</h4>
                        <div className="node-inspector__info">
                            {Object.entries(data.extra_info).map(([key, value]) => (
                                <div className="node-inspector__row" key={key}>
                                    <span className="node-inspector__key">{key}</span>
                                    <code className="node-inspector__value">
                                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                    </code>
                                </div>
                            ))}
                        </div>
                    </section>
                )}
            </div>
        </aside>
    );
}
