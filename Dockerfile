FROM python:3.13

WORKDIR /serve

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV MODEL_PATH=/serve/models/model.json

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
