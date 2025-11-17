FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x run_script.sh

ENTRYPOINT ["./run_script.sh"]

