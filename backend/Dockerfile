FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV / MediaPipe 需要 libGL、glib；build-essential 用來編譯 torchreid 的 Cython 模組
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU 版 PyTorch（CUDA 12.6；wheel 內含 CUDA runtime，主機只需 NVIDIA 驅動）
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

# torchreid（OSNet 跨鏡頭 Re-ID）：PyPI 上沒有，從 GitHub 原始碼安裝
RUN pip install cython tensorboard h5py yacs gdown \
    && pip install --no-build-isolation https://github.com/KaiyangZhou/deep-person-reid/archive/refs/heads/master.zip

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
