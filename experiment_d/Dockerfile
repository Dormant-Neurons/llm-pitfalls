FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3.12 python3.12-venv python3-pip
RUN apt-get install -y texlive texlive-latex-extra

COPY requirements.txt .

RUN pip install uv
RUN uv pip install -r requirements.txt --system

COPY . .

ENTRYPOINT ["python3.10", "run_experiment.py"]
