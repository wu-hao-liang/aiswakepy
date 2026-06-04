# Deployment

The app can run in Docker with Waitress for local testing and Gunicorn on the VPS.

## Local WSL Test

This machine currently has Docker but not the `docker compose` plugin. Use plain Docker:

```bash
docker build -t aiswakepy:local .
docker run --rm -p 8050:8050 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/server_config.json:/app/server_config.json:ro" \
  aiswakepy:local .venv/bin/python dash_app.py
```

Open `http://localhost:8050`.

If the Compose plugin is installed, the equivalent command is:

```bash
docker compose -f docker-compose.local.yml up --build
```

The local compose file mounts:
- `./data` to `/app/data`
- `./output` to `/app/output`
- `./server_config.json` read-only

## VPS

On the Ubuntu VPS, install Docker with the Compose plugin, then:

```bash
git clone git@github.com:wu-hao-liang/aiswakepy.git
cd aiswakepy
git checkout feature/public-web-app
docker compose -f docker-compose.prod.yml up -d --build
```

Without Compose:

```bash
docker build -t aiswakepy:prod .
docker volume create aiswakepy-data
docker volume create aiswakepy-output
docker run -d --name aiswakepy --restart unless-stopped -p 8050:8050 \
  -e GUNICORN_WORKERS=2 \
  -e GUNICORN_THREADS=4 \
  -e GUNICORN_TIMEOUT=300 \
  -v aiswakepy-data:/app/data \
  -v aiswakepy-output:/app/output \
  aiswakepy:prod
```

The production compose file uses Gunicorn:

```bash
.venv/bin/gunicorn dash_app:server --bind 0.0.0.0:8050
```

It stores runtime files in Docker volumes:
- `aiswakepy-data`
- `aiswakepy-output`

Put Nginx/Caddy in front of the container for HTTPS and public domain routing.
