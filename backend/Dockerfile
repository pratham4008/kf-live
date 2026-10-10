FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY research/ ./research/
# SQLite lives on a mounted volume so results survive restarts and redeploys
VOLUME ["/data"]
ENV DB_PATH=/data/live.db
EXPOSE 8000
CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
