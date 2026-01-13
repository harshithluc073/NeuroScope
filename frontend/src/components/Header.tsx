import { useState, useCallback, useEffect } from 'react';
import { useGraphStore } from '../store/graphStore';

export function Header() {
    const { nodes, graphData, clearGraph, setSearchQuery, searchQuery } = useGraphStore();
    const [isExporting, setIsExporting] = useState(false);

    const modelName = (graphData?.metadata?.model_class as string) || 'No Model';
    const framework = (graphData?.metadata?.framework as string) || '';
    const nodeCount = nodes.length;

    // Export to PNG
    const exportPng = useCallback(async () => {
        const viewport = document.querySelector('.react-flow__viewport') as HTMLElement;
        if (!viewport || isExporting) return;

        setIsExporting(true);
        try {
            // Dynamic import to avoid issues
            const { toPng } = await import('html-to-image');
            const dataUrl = await toPng(viewport, {
                backgroundColor: '#0d1117',
                pixelRatio: 2,
            });

            const link = document.createElement('a');
            link.download = `neuroscope-${modelName}-${Date.now()}.png`;
            link.href = dataUrl;
            link.click();
        } catch (error) {
            console.error('Export PNG failed:', error);
        } finally {
            setIsExporting(false);
        }
    }, [modelName, isExporting]);

    // Export to SVG
    const exportSvg = useCallback(async () => {
        const viewport = document.querySelector('.react-flow__viewport') as HTMLElement;
        if (!viewport || isExporting) return;

        setIsExporting(true);
        try {
            const { toSvg } = await import('html-to-image');
            const dataUrl = await toSvg(viewport, {
                backgroundColor: '#0d1117',
            });

            const link = document.createElement('a');
            link.download = `neuroscope-${modelName}-${Date.now()}.svg`;
            link.href = dataUrl;
            link.click();
        } catch (error) {
            console.error('Export SVG failed:', error);
        } finally {
            setIsExporting(false);
        }
    }, [modelName, isExporting]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ctrl/Cmd + F: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                const searchInput = document.getElementById('graph-search');
                searchInput?.focus();
            }
            // Ctrl/Cmd + E: Export PNG
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                exportPng();
            }
            // Escape: Clear search
            if (e.key === 'Escape') {
                setSearchQuery('');
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [exportPng, setSearchQuery]);

    return (
        <header className="header">
            <div className="header__brand">
                <span className="header__logo">N</span>
                <h1 className="header__title">NeuroScope</h1>
            </div>

            <div className="header__search">
                <input
                    id="graph-search"
                    type="text"
                    placeholder="Search nodes... (Ctrl+F)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="header__search-input"
                />
                {searchQuery && (
                    <button
                        className="header__search-clear"
                        onClick={() => setSearchQuery('')}
                        title="Clear search (Esc)"
                    >
                        ×
                    </button>
                )}
            </div>

            <div className="header__info">
                {graphData && (
                    <>
                        <span className="header__model">{modelName}</span>
                        <span className="header__divider">|</span>
                        <span className="header__framework">{framework}</span>
                        <span className="header__divider">|</span>
                        <span className="header__nodes">{nodeCount} nodes</span>
                    </>
                )}
            </div>

            <div className="header__actions">
                <button
                    className="header__button"
                    onClick={() => void exportPng()}
                    disabled={nodes.length === 0 || isExporting}
                    title="Export as PNG (Ctrl+E)"
                >
                    PNG
                </button>
                <button
                    className="header__button"
                    onClick={() => void exportSvg()}
                    disabled={nodes.length === 0 || isExporting}
                    title="Export as SVG"
                >
                    SVG
                </button>
                <button
                    className="header__button header__button--secondary"
                    onClick={clearGraph}
                    disabled={nodes.length === 0}
                    title="Clear graph"
                >
                    Clear
                </button>
            </div>
        </header>
    );
}
