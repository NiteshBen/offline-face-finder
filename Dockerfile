# Base image: Official lightweight Python runtime (compatible with AMD64 & ARM64)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install Linux system dependencies required by OpenCV DNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-downloaded offline models
COPY models/ ./models/
COPY database.py face_engine.py scanner.py app_gui.py app_web.py main.py ./

# Create mount points for external photos and database persistence
RUN mkdir -p /photos /data
ENV DATABASE_DIR=/data

# Expose Streamlit web interface port
EXPOSE 8501

# Run the offline web interface
CMD ["streamlit", "run", "app_web.py"]
