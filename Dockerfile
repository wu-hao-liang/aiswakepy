FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# fiona's GDAL wheel (and every fiona C-extension) dynamically links the system
# libexpat.so.1, which the manylinux wheel deliberately does NOT bundle. The
# python:3.12-slim base builds Python with a vendored expat, so it ships no
# system libexpat1 — without it, shapefile reading (coastline/land preview and
# land masking) fails with "libexpat.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN uv sync --no-dev

RUN mkdir -p /data/uploads /app/output

EXPOSE 8050

CMD ["sh", "-c", ".venv/bin/gunicorn dash_app:server --bind 0.0.0.0:8050 --workers ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-300}"]
