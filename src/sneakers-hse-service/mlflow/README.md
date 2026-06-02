# MLflow stack

This folder contains a local Docker Compose stack for MLflow with:

- MLflow Tracking Server
- PostgreSQL as the MLflow backend store
- MinIO as S3-compatible artifact storage
- MinIO Client (`mc`) for automatic bucket creation

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose v2
- Internet access to pull images from `ghcr.io`, `quay.io`, and Docker Hub

Check that Docker is running:

```bash
docker compose version
docker ps
```

## Start From Scratch

Go to this directory:

```bash
cd src/sneakers-hse-service/mlflow
```

Create `.env`:

```bash
cat > .env <<'EOF'
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow

MLFLOW_PORT=5050
S3_API_PORT=9000
S3_UI_PORT=9001

MLFLOW_S3_ENDPOINT_URL=http://minio:9000
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password
DEFAULT_BUCKET_NAME=mlflow-bucket
EOF
```

Create local data directories used by bind volumes:

```bash
mkdir -p data/minio_data data/mlflow
```

Build and start services:

```bash
docker compose up -d --build
```

Check that everything is running:

```bash
docker compose ps
```

Expected services:

- `mlflow-service` on `http://localhost:5050`
- `minio` API on `http://localhost:9000`
- `minio` console on `http://localhost:9001`
- `postgres` on `localhost:5432`
- `minio_mc`, which creates the `mlflow-bucket` bucket

MinIO console credentials:

- user: `admin`
- password: `password`

## Use MLflow From Python

Install MLflow in your Python environment if needed:

```bash
pip install mlflow boto3
```

Point your code to the local tracking server:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5050")

with mlflow.start_run():
    mlflow.log_param("example_param", "value")
    mlflow.log_metric("example_metric", 1.0)
```

Artifacts logged through MLflow are stored in MinIO in the `mlflow-bucket` bucket.

## Stop Services

Stop containers but keep local data:

```bash
docker compose down
```

Stop containers and remove Docker volumes:

```bash
docker compose down -v
```

If you also want to delete local MinIO and MLflow data, remove the `data/` directory manually.

## Troubleshooting

If Docker cannot mount `mlflow_minio_data`, make sure the local directory exists:

```bash
mkdir -p data/minio_data
docker compose up -d --build
```

If image pulling fails with `TLS handshake timeout`, rerun the command. Docker usually continues from already downloaded layers:

```bash
docker compose pull mlflow-service
docker compose up -d --build
```

If ports are already used, edit `.env` and change:

- `MLFLOW_PORT`
- `S3_API_PORT`
- `S3_UI_PORT`

Then restart:

```bash
docker compose down
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f
```

View logs for one service:

```bash
docker compose logs -f mlflow-service
```
