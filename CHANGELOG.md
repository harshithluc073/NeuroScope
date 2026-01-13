# Changelog

All notable changes to NeuroScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-13

### Added
- **PyTorch Tracer**: Full support for tracing PyTorch models using forward hooks
  - Automatic capture of tensor shapes, dtypes, and devices
  - NaN/Inf detection in outputs
  - Hierarchical node depth tracking
  - Edge creation from tensor flow
  
- **TensorFlow Tracer**: Support for Keras models
  - Layer wrapping with call hooks
  - Recursive sublayer support
  - Configuration extraction
  
- **JAX Tracer**: Support for JAX functions
  - jaxpr introspection and parsing
  - Works with Flax/Haiku modules
  
- **WebSocket Server**: Real-time graph streaming
  - Optional API key authentication
  - CORS origin validation
  - Rate limiting (100 msg/sec)
  - Batched broadcasting for performance
  
- **React Frontend**: Interactive graph visualization
  - React Flow with Dagre layout
  - Custom node components with color coding
  - Node inspector panel
  - Search and filter functionality
  - PNG/SVG export
  - Keyboard shortcuts (Ctrl+F, Ctrl+E, Escape)
  
- **Error Handling**: Production-ready robustness
  - Hooks never crash training loops
  - MAX_NODES limit (10,000) for memory protection
  - React ErrorBoundary for frontend recovery
  
- **Docker Support**: Multi-stage Dockerfile for deployment
- **CI/CD**: GitHub Actions for testing and PyPI publishing

### Security
- Optional API key authentication via `NEUROSCOPE_API_KEY`
- CORS origin validation
- Rate limiting per client

[Unreleased]: https://github.com/neuroscope/neuroscope/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/neuroscope/neuroscope/releases/tag/v0.1.0
