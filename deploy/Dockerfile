FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build Qallow
RUN cmake -S . -B build -G Ninja && \
    cmake --build build --target qallow_unified -j

# Install Python dependencies for dashboard
RUN pip install Flask Flask-CORS

# Expose ports
EXPOSE 5000 8080

# Default command
CMD ["./build/qallow_unified", "run"]

