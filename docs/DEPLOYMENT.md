# Deployment

The app can run in Docker with Waitress for local testing and Gunicorn on the VPS.

## Local WSL Test

This machine currently has Docker but not the `docker compose` plugin. Use plain Docker:

```bash
docker build -t aiswakepy:local .
docker run --rm -p 8050:8050 \
  -e DATA_ROOT=/data \
  -e UPLOAD_FOLDER=/data/uploads \
  -v "$PWD/data:/data" \
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
- `./data` to `/data`
- `./output` to `/app/output`
- `./server_config.json` read-only

## VPS

On the Ubuntu VPS, install Docker with the Compose plugin, then:

```bash
git clone git@github.com:wu-hao-liang/aiswakepy.git
cd aiswakepy
git checkout feature/public-web-app
cp .env.example .env
sudo mkdir -p /opt/apps/aiswakepy/data
sudo chown -R "$USER":"$USER" /opt/apps/aiswakepy/data
docker compose -f docker-compose.prod.yml up -d --build
```

Edit `.env` before starting the service if the host port or Gunicorn settings need to change.
The production defaults still map `/opt/apps/aiswakepy/data` on the VPS to `/data` inside the container for future account-backed storage. Anonymous public uploads are stored below `UPLOAD_FOLDER=/data/uploads` as temporary page sessions and are cleaned up after `SESSION_TTL_SECONDS`.

Without Compose:

```bash
docker build -t aiswakepy:prod .
docker volume create aiswakepy-output
cp .env.example .env
sudo mkdir -p /opt/apps/aiswakepy/data
sudo chown -R "$USER":"$USER" /opt/apps/aiswakepy/data
docker run -d --name aiswakepy --restart unless-stopped -p 8050:8050 \
  --env-file .env \
  -v /opt/apps/aiswakepy/data:/data \
  -v aiswakepy-output:/app/output \
  aiswakepy:prod
```

The production compose file uses Gunicorn:

```bash
.venv/bin/gunicorn dash_app:server --bind 0.0.0.0:8050
```

Data and uploads are stored at `/opt/apps/aiswakepy/data` on the VPS. Generated output is stored in the `aiswakepy-output` Docker volume.
Keep `GUNICORN_WORKERS=1`: the current app stores pipeline state and map caches in process memory. Use `GUNICORN_THREADS` for request concurrency.

## Verify

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail=100 aiswakepy
curl -I http://127.0.0.1:8050/
```

The service should report healthy and `curl` should return HTTP 200.

Verify the bind mount:

```bash
docker compose -f docker-compose.prod.yml exec aiswakepy \
  .venv/bin/python -c "from pathlib import Path; Path('/data/volume-test.txt').write_text('ok\n')"
cat /opt/apps/aiswakepy/data/volume-test.txt
rm /opt/apps/aiswakepy/data/volume-test.txt
```

Put Nginx/Caddy in front of the container for HTTPS and public domain routing.
