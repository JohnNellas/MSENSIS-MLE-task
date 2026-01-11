# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV/PyTorch image handling
# (libgl1-mesa-glx is a common missing library for cv2/images)
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# We don't specify CMD here because we will override it in docker-compose