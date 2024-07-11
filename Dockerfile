FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Installing dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    git \
    wget \
    python3-dev \
    python3-pip \
    unzip \
    nano \
    cmake \
    curl \
    lsof \
    net-tools && \
    apt-get clean

RUN python3 -m pip install pip --upgrade

# Set the working directory
WORKDIR /app

# Copy requirements.txt first for caching
COPY inference/requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY inference /app/inference

# Specify the command to run your application
CMD ["bash"]
