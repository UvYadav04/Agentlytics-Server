
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libgl1 libglib2.0-0 \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/requirements.txt shared/requirements.txt
COPY analyzerEngine/requirements.txt analyzerEngine/requirements.txt
COPY worker_service/requirements.txt worker_service/requirements.txt


ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r worker_service/requirements.txt

COPY shared/ shared/
COPY analyzerEngine/ analyzerEngine/
COPY worker_service/ worker_service/

ENV PYTHONUNBUFFERED=1

CMD ["arq", "worker_service.worker.WorkerSettings"]
