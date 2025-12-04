COPY requirements.txt /tmp/requirements.txt
# Использование образа с Mamba/Conda
FROM mcr.microsoft.com/devcontainers/miniconda:latest

# Установка системных зависимостей (требуется для компиляции некоторых пакетов, например, PyTorch3D)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    # Установка других пакетов, если они необходимы для PyTorch3D или SMAL
    
# Переход в рабочую директорию
WORKDIR /workspaces/inferno

# Копирование файла требований PyTorch (мы используем более свежую версию 3.9)
COPY requirements39_torch.txt /tmp/requirements.txt

# Установка Conda окружения
# В этом проекте рекомендуется использовать Mamba (ускоренная версия Conda)
# Установка зависимостей из requirements.txt
RUN mamba create -n inferno_env python=3.9 -y \
    && echo "conda activate inferno_env" >> ~/.bashrc \
    && mamba install -n inferno_env --file /tmp/requirements.txt -y \
    && mamba clean -afy
    
# Активация среды для последующих команд
SHELL ["conda", "run", "-n", "inferno_env", "/bin/bash", "-c"]

# Установка самого проекта (если это пакет Python)
# Если есть файл setup.py, используйте:
# RUN pip install -e .

# Определение пользователя для Codespaces
USER vscode
