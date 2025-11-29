# Базовый образ Ubuntu + Python
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Создаём пользователя vscode
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspace

# Копируем requirements
COPY requirements.txt /workspace/requirements.txt

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
