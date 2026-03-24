# Use PyTorch with CUDA support for GPU inference
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Install git (needed by huggingface_hub to clone model repos)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY app.py .

# Expose port 7860 to the outside world
EXPOSE 7860

# Define environment variables
ENV PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=false
ENV LORA_ALLOW_REMOTE=false
ENV HF_HOME=/tmp/hf-cache
ENV TRANSFORMERS_CACHE=/tmp/hf-cache

# Run app.py when the container launches
CMD ["python", "-u", "app.py"]
