# Build context must be Server/ (not api_service/), since this needs
# shared/ as a sibling: docker build -f api_service/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

COPY shared/requirements.txt shared/requirements.txt
COPY api_service/requirements.txt api_service/requirements.txt
RUN pip install --no-cache-dir -r api_service/requirements.txt

COPY shared/ shared/
COPY api_service/ api_service/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
