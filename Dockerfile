# COM7019 stock forecasting — CPU training image (GPU: use Colab or nvidia/cuda base)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

COPY Stock_Price_Data_[3921].csv ./data/
COPY _code/COM7019_25199053.ipynb ./notebooks/

RUN mkdir -p /app/outputs

VOLUME ["/app/outputs"]

CMD ["python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", \
     "notebooks/COM7019_25199053.ipynb", "--output", "COM7019_25199053_executed.ipynb"]
