# Используем официальный образ Dev Containers с Python 3.10
# (вы можете выбрать другую версию, если ваш проект требует этого)
FROM mcr.microsoft.com/devcontainers/python:3.10

# Установка системных зависимостей
# В проектах NLP иногда нужны определенные системные пакеты
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    build-essential \
    # Добавьте сюда другие пакеты, если они нужны для PyTorch или других библиотек

# Установка зависимостей Python
# Копируем requirements.txt и устанавливаем библиотеки
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Установка самого проекта Inferno
# Используется, если проект содержит модули, которые нужно импортировать
# RUN pip install -e .

# Определение пользователя для Codespaces
USER vscode
