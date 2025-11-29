# Базовый образ с CUDA для GPU
FROM nvidia/cuda:12.2.1-cudnn8-runtime-ubuntu22.04

# Установка базовых пакетов
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install numpy==1.26.4 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    opencv-python matplotlib Pillow scikit-image tqdm numba gradio

WORKDIR /workspaces/inferno
