FROM python:3.11-alpine
WORKDIR /app
RUN apk add --no-cache \
    gcc musl-dev linux-headers g++ libffi-dev curl \
    gfortran openblas-dev lapack-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/models

ENV PYTHONPATH=/app
EXPOSE 5000

CMD ["python", "/app/api/optimization_service.py"]