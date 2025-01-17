FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    vim && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 && \
    rm -rf /var/lib/apt/lists/*
CMD ["bash"]                    