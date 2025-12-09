FROM python:3.11-alpine

WORKDIR /app

# Install build dependencies for numpy/pandas and curl for healthchecks
RUN apk add --no-cache gcc musl-dev linux-headers g++ libffi-dev curl

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire ml directory structure
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 5000

# Run the optimization service
CMD ["python", "/app/api/optimization_service.py"]