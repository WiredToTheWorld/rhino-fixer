# Container for hosting Rhino Fixer on Google Cloud Run (works on any Docker host).
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
# Cloud Run sends traffic to $PORT (8080 by default) and caps request bodies at
# 32 MB, so uploads are limited to 32 MB. /tmp lives in memory on Cloud Run, so
# finished jobs are cleared after 2 hours to keep memory free.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    HOSTED=1 \
    MAX_UPLOAD_MB=32 \
    MAX_CONCURRENT_JOBS=2 \
    JOB_TTL_HOURS=2 \
    JOBS_DIR=/tmp/rhino-fixer-jobs
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user server.py .
COPY --chown=user rhinofixer ./rhinofixer
COPY --chown=user web ./web

EXPOSE 8080
CMD ["python", "server.py"]
