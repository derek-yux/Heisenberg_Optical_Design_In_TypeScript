FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose API port
EXPOSE 5000

# Run the optimization service
CMD ["python", "api/optimization_service.py"]