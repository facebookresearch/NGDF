#!/bin/bash
docker build -t ros-base:v1.0 --build-arg BUILD_USER_ID=$(id -u) --build-arg BUILD_GROUP_ID=$(id -g) dockerfiles/base
docker build --build-arg SSH_PRIVATE_KEY="$(cat $HOME/.ssh/id_rsa)" --build-arg SSH_PUBLIC_KEY="$(cat $HOME/.ssh/id_rsa.pub)" -t ros-desktop:v1.0 dockerfiles/desktop