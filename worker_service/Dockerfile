# Build context must be Server/: docker build -f worker_service/Dockerfile .
#
# See engine_bootstrap.py's PARQUET_ROOT comment before running this in
# production - the Tabular Agent's Docker sandbox needs the host's Docker
# socket AND a host-resolvable bind mount path, which containerizing this
# service does not get you for free. Simplest reliable option: run
# worker_service as a bare process on a VM with Docker installed, not as a
# container. This Dockerfile is provided for platforms/setups where you've
# solved that (e.g. a single-host docker-compose deployment where the bind
# mount path matches on both sides).
FROM python:3.12-slim

# docker CLI client (talks to the mounted host socket) + build tools some
# analyzerEngine deps (docling, chromadb) need at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/requirements.txt shared/requirements.txt
COPY analyzerEngine/requirements.txt analyzerEngine/requirements.txt
COPY worker_service/requirements.txt worker_service/requirements.txt

# docling's layout/OCR/chart-extraction models pull in torch (and, via
# transformers' image processing, torchvision) transitively. The default PyPI
# wheels drag along the full CUDA runtime (nvidia-cublas/cudnn/nccl/... several
# GB) even though this container has no GPU. Install CPU-only torch AND
# torchvision together, from the same index, in one command - torchvision's
# compiled ops must match the exact torch build they're loaded against, or you
# get "operator torchvision::nms does not exist" at import time. Installing
# them separately (or torch here + torchvision resolved later from PyPI by
# docling's own dependency chain) breaks that pairing.
# (The reranker itself no longer needs torch - it now calls the Hugging Face
# Inference API instead of running a local sentence-transformers model.)
#
# Also make the CPU index available for the *rest* of the install below - if
# docling's own version constraints don't match what we just installed, pip
# would otherwise silently re-resolve torch/torchvision from default PyPI
# (bringing the CUDA wheels back) instead of picking a compatible CPU build.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r worker_service/requirements.txt

COPY shared/ shared/
COPY analyzerEngine/ analyzerEngine/
COPY worker_service/ worker_service/

ENV PYTHONUNBUFFERED=1

CMD ["arq", "worker_service.worker.WorkerSettings"]
