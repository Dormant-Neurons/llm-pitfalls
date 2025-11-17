FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3.12 python3.12-venv python3-pip git curl
RUN pip install uv
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN git clone https://github.com/LostOxygen/llm-confidentiality
WORKDIR "./llm-confidentiality"
RUN uv pip install -r requirements.txt --system

COPY . .
RUN chmod +x add_key.sh
RUN ./add_key.sh

ENTRYPOINT ["python3.10", "attack.py"]