#!/bin/bash

# get directory of current file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR_NAME=${PWD##*/} # name of file dir
CONDA_ROOT=$(conda info --base)

# run docker container
# Mount code directory into both /workspace 
# and original path on host so pip -e installs don't break
docker run \
    -v $CONDA_ROOT:$CONDA_ROOT \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DIR:/workspace/$DIR_NAME  \
    -v $DIR:$DIR \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e PATH=$CONDA_ROOT/bin:$PATH \
    -it ngdf:latest bash
