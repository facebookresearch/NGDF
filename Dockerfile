FROM ubuntu:20.04

RUN apt update
RUN apt install -y libsuitesparse-dev

RUN mkdir -p /workspace/NGDF
WORKDIR /workspace/NGDF