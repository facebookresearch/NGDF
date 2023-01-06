##
## How do I select the base image?
## Check the host nvidia version by running: nvcc --version
## That will give you your host's cuda version.
## Then try to find one that matches from the release notes
## of nVidia's images: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-12.html#rel_21-12
## Finally, the tag in: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow/tags
## You will have to also change the installation of pytorch so that the cuda version also fits
##
## For the current version, the host, the docker image and pytorch are CUDA 11.5
##
## First you can try the base by using: docker run --runtime=nvidia --rm nvcr.io/nvidia/tensorflow:21.12-tf2-py3 /bin/bash
##
FROM nvcr.io/nvidia/tensorflow:21.12-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'Etc/UTC' > /etc/timezone && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    apt-get autoclean

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 && \
    apt-get autoclean
    
##
## What's at the end of this line "cu115" should be consistent with both:
##  (1) The cuda version in the host that shall be checked with "nvcc --version"
##  (2) The cuda version in the base image which you can check in 
##      https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-12.html#rel_21-12
RUN pip install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG BUILD_GROUP_ID
ARG BUILD_USER_ID

# Create 'ros' user with sudo powers
ARG USERNAME=ros
RUN groupadd --gid $BUILD_GROUP_ID $USERNAME && \
    useradd -s /bin/bash --uid $BUILD_USER_ID --gid $BUILD_GROUP_ID -m $USERNAME && \
    mkdir -p /etc/sudoers.d && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git vim tree net-tools \
    python3-pip \
    apt-file \
    graphicsmagick-libmagick-dev-compat \
    libceres-dev \
    libzmq3-dev \
    libboost-dev \
    jq \
    wget \
    python3-tk \
    curl \
    libssl-dev \
    libusb-1.0-0-dev \
    usb-1.0-0 \
    libudev-dev \
    udev \
    pkg-config  \
    libgtk-3-dev \
    freeglut3 \
    freeglut3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    fswebcam \
    xdg-utils && \
    apt-get autoclean

RUN apt-get update && \
    apt install -y lsb-release  && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'  && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get autoclean
    
RUN apt-get update && \
    apt install -y ros-noetic-desktop-full python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    apt-get autoclean

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall

USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo mkdir /etc/ros && \
    sudo chown -R ros:ros /etc/ros && \
    sudo chown -R ros:ros /usr/local/lib/python3.8/dist-packages

RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && \
  sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/ && \
  sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' && \
  rm -f packages.microsoft.gpg

RUN sudo apt update && \
    sudo apt install --no-install-recommends -y apt-transport-https  code ninja-build && \
    sudo apt-get autoclean
