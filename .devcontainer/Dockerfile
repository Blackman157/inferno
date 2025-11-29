# Используем аргумент VARIANT для версии Ubuntu
ARG VARIANT=22.04
FROM nvidia/cuda:12.2.1-cudnn8-runtime-ubuntu${VARIANT}

# Основные утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl wget build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Установка последней версии pip
RUN python3 -m pip install --upgrade pip

# Создание рабочей директории
WORKDIR /workspace

# Копирование файлов проекта (если нужно)
# COPY . /workspace

# По желанию, установка зависимостей проекта
# RUN pip install -r requirements.txt
