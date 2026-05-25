FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md License ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev,experiment,report]"

CMD ["cl-bench", "run", "--config", "configs/smoke.yaml", "--method", "baseline", "--epochs", "1", "--device", "cpu", "--output-dir", "/tmp/cl-bench-runs"]
