# NeuroScope Docker Image
# Multi-stage build for Python backend + React frontend

# ====================
# Stage 1: Build Frontend
# ====================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source
COPY frontend/ ./

# Build production bundle
RUN npm run build

# ====================
# Stage 2: Python Runtime
# ====================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy Python package
COPY pyproject.toml README.md ./
COPY neuroscope/ ./neuroscope/
COPY examples/ ./examples/

# Install Python package
RUN pip install --no-cache-dir -e ".[pytorch]"

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV NEUROSCOPE_HOST=0.0.0.0
ENV NEUROSCOPE_PORT=8765

# Expose ports
EXPOSE 8765
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

# Default command - start a simple HTTP server for frontend + example
CMD ["python", "-c", "import http.server; import socketserver; handler = http.server.SimpleHTTPRequestHandler; with socketserver.TCPServer(('', 3000), handler) as httpd: print('Frontend at http://localhost:3000'); httpd.serve_forever()"]
