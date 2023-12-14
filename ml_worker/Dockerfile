FROM python:3.10-rc-slim-buster

COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

ENV JAVA_HOME /usr/local/openjdk-8
ENV PATH $PATH:$JAVA_HOME/bin

RUN apt-get update && \
    yes | apt-get upgrade && \
    apt-get -y install --no-install-recommends \
                       build-essential \
                       make \
                       gcc \
                       libc-dev \
                       musl-dev \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       tesseract-ocr \
                       libtesseract-dev \
                       libgl1-mesa-glx \
                       libglib2.0-0 \
                       wget \
                       zlib1g-dev \
                       libreoffice-common \
                       wkhtmltopdf \
                       libreoffice-writer \
		               git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# RUN pip install -q torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git'

ADD . /ml_worker
WORKDIR /ml_worker
RUN mkdir /nonexistent
RUN chmod -R 777 /nonexistent

# ENTRYPOINT celery -A tasks worker --loglevel=info
ENTRYPOINT celery -A tasks worker -P solo --loglevel=info
# ENTRYPOINT celery -A tasks worker -P eventlet --concurrency=4 --loglevel=info

#-Q celery_worker
#ENTRYPOINT ['celery','-A','test_celery', 'worker', '--loglevel=info']

