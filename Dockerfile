# Use the slim Debian image which provides wide wheel compatibility (glibc)
FROM python:3.10-slim 

# Install dependencies needed for soundfile to work natively
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install `uv` for ultra-fast dependency resolution and installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Use `uv` to create a virtual environment and install dependencies.
RUN uv venv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY api.py vics_ir_generator.py cuki_ir_core.py ./
COPY static/ ./static/

# Expose the API port
EXPOSE 8000

# Run the FastAPI server via Uvicorn (it's installed globally in the container's python by uv)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
