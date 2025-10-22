FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Clean build directory if it exists
RUN rm -rf build

# Build Qallow
RUN cmake -S . -B build -G Ninja && \
    cmake --build build --target qallow_unified -j

# Runtime stage
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/build/qallow_unified ./build/

# Copy source code for dashboard
COPY ui ./ui/
COPY src ./src/
COPY include ./include/

# Install Python dependencies for dashboard
RUN pip install Flask Flask-CORS

# Expose ports
EXPOSE 5000 8080

# Default command
CMD ["./build/qallow_unified", "run"]

