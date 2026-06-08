FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir aiogram aiohttp

COPY bot.py .

CMD ["python", "bot.py"]
