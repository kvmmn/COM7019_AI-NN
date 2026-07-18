FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e . \
    && pip install jupyter nbconvert

COPY ["data/Stock_Price_Data_[3921].csv", "./Stock_Price_Data_[3921].csv"]
COPY notebook/COM7019_25199053.ipynb ./notebook/

RUN mkdir -p /app/output

CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", \
     "notebook/COM7019_25199053.ipynb", \
     "--output", "/app/output/COM7019_25199053_executed.ipynb", \
     "--ExecutePreprocessor.timeout=7200"]
