FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system --gid 10001 materialgraph \
    && useradd --system --uid 10001 --gid materialgraph --home-dir /app materialgraph

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /app/data/runtime \
    && chown -R materialgraph:materialgraph /app

USER 10001:10001
CMD ["material-graph-knowledge", "--help"]
