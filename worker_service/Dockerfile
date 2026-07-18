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
# analyzerEngine deps (docling, sentence-transformers, chromadb) need at
# install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/requirements.txt shared/requirements.txt
COPY analyzerEngine/requirements.txt analyzerEngine/requirements.txt
COPY worker_service/requirements.txt worker_service/requirements.txt
RUN pip install --no-cache-dir -r worker_service/requirements.txt

COPY shared/ shared/
COPY analyzerEngine/ analyzerEngine/
COPY worker_service/ worker_service/

ENV PYTHONUNBUFFERED=1

CMD ["arq", "worker_service.worker.WorkerSettings"]
