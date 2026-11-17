# syntax=docker/dockerfile:1

# ---- build stage: resolve and install into an isolated virtualenv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# ---- runtime stage: no compilers, no pip cache, non-root user --------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    GQB_DATA_DIR=/data \
    GLIMPSE_LIVE_TRADING=false

RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /data/recordings \
    && chown -R bot:bot /data

COPY --from=builder /opt/venv /opt/venv

USER bot
WORKDIR /home/bot

# /data holds the SQLite ledger and NDJSON audit log; /data/recordings the
# MarketRecorder NDJSON files. Mount volumes here to keep them across restarts.
VOLUME ["/data", "/data/recordings"]

ENTRYPOINT ["glimpse-quant-bot"]
# Paper trading. Real orders additionally need `run --live` AND GLIMPSE_LIVE_TRADING=true.
CMD ["run"]
