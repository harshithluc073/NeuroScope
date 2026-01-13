import type { ConnectionStatus as Status } from '../store/graphStore';

interface ConnectionStatusProps {
    status: Status;
}

const STATUS_CONFIG = {
    disconnected: { label: 'Disconnected', color: 'var(--status-disconnected)' },
    connecting: { label: 'Connecting...', color: 'var(--status-connecting)' },
    connected: { label: 'Connected', color: 'var(--status-connected)' },
    error: { label: 'Connection Error', color: 'var(--status-error)' },
};

export function ConnectionStatus({ status }: ConnectionStatusProps) {
    const config = STATUS_CONFIG[status];

    return (
        <div className="connection-status" title={`Server: ${config.label}`}>
            <span
                className="connection-status__dot"
                style={{ backgroundColor: config.color }}
            />
            <span className="connection-status__label">{config.label}</span>
        </div>
    );
}
