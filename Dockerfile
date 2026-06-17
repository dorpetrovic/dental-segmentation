# Dental Teeth Segmentation — Torchvision Mask R-CNN
# Base: PyTorch 2.0 with CUDA 11.8

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL description="Mask R-CNN (torchvision) for dental tooth instance segmentation"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        git \
        wget \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Verify GPU and torchvision ────────────────────────────────────────────────
RUN python3 -c "\
import torch, torchvision; \
print('PyTorch:    ', torch.__version__); \
print('Torchvision:', torchvision.__version__); \
print('CUDA:       ', torch.cuda.is_available())"

# ── Copy project source ───────────────────────────────────────────────────────
COPY . /app/

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p outputs/logs \
             outputs/results/maskrcnn_torch \
             outputs/visualizations

# ── Environment variables ─────────────────────────────────────────────────────
# dont write .pyc files inside condainer - keep image clean
ENV PYTHONDONTWRITEBYTECODE=1
# logging appears immediatly in docker logs, not buffered
ENV PYTHONUNBUFFERED=1
ENV MODEL_WEIGHTS=/app/outputs/results/maskrcnn_torch/best.pth

# ── Expose API port ───────────────────────────────────────────────────────────
#FastAPI
EXPOSE 8000

#Gradio
EXPOSE 7860

# ── Default command — API server ──────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -f Dockerfile -t dental-maskrcnn-torch:latest .
#
# Train:
#   docker run --gpus all \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/outputs:/app/outputs \
#       -e PYTHONUNBUFFERED=1 \
#       --shm-size=4gb \
#       --name maskrcnn_training \
#       dental-maskrcnn-torch:latest \
#       python models/teeth_segmentation.py train

# Evaluate:
#   docker run --gpus all \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/outputs:/app/outputs \
#       dental-maskrcnn-torch:latest \
#       python models/teeth_segmentation_torch.py evaluate
#
# Predict:
#   docker run --gpus all \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/outputs:/app/outputs \
#       dental-maskrcnn-torch:latest \
#       python models/teeth_segmentation_torch.py predict \
#           --image /app/data/test/012.jpg
#
# FastAPI server:
#   docker run --gpus all -p 8000:8000 \
#       -v $(pwd)/outputs:/app/outputs \
#       -v $(pwd)/data:/app/data \
#       dental-maskrcnn-torch:latest
# 
# Gradio demo:
#   docker run --gpus all -p 7860:7860 \
#        -v $(pwd)/outputs:/app/outputs \
#        -v $(pwd)/data:/app/data \
#        dental-maskrcnn-torch:latest \
#        python app/gradio_demo.py 

