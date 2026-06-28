FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for soundfile
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default command
CMD ["python", "examples/01_detection_demo.py"]
