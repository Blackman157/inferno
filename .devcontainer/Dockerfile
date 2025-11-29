# Базовый образ Python 3.11
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую папку
WORKDIR /workspace

# Установка Python-библиотек
# Используем CPU-версию torch
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python matplotlib pillow numpy scikit-image tqdm

# Клонируем inferno
RUN git clone https://github.com/YadiraF/Inferno.git /workspace/inferno

# Установка зависимостей inferno
RUN pip install --no-cache-dir -r /workspace/inferno/requirements.txt

# По умолчанию открываем рабочую папку
WORKDIR /workspace/inferno
