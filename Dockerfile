FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY . .

RUN uv sync --no-dev

RUN mkdir -p /app/data /app/output

EXPOSE 8050

CMD ["sh", "-c", ".venv/bin/gunicorn dash_app:server --bind 0.0.0.0:8050 --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-300}"]
