FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install python-dotenv

RUN chmod +x run_experiments.sh

ENTRYPOINT ["./run_experiments.sh", "openai,anthropic,local"]

