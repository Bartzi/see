# Use an official Python runtime as a parent image
ARG FROM_IMAGE=nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM ${FROM_IMAGE}

# install opencv for python 3
RUN apt-get update && \
  apt-get install -y \
  build-essential \
  git \
  libasound2-dev \
  libavformat-dev \
  libcanberra-gtk3-module \
  libgtk-3-dev \
  libjasper-dev \
  libjpeg-dev \
  libpng-dev \
  libpq-dev \
  libswscale-dev \
  libtbb-dev \
  libtbb2 \
  libtiff-dev \
  pkg-config \
  python3 \
  python3-numpy \
  python3-pip \
  unzip \
  wget \
  yasm

# Set the working directory to /app
WORKDIR /app

ARG NCCL_NAME=nccl-repo-ubuntu1604-2.1.15-ga-cuda9.1_1-1_amd64.deb
COPY ${NCCL_NAME} /app
RUN dpkg -i ${NCCL_NAME}
RUN apt-get update && apt-get install -y libnccl2 libnccl-dev

COPY requirements.txt /app/
RUN pip3 install -v --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV NAME See
