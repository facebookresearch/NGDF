#!/bin/bash
sudo apt-get uninstall -y docker docker-engine docker.io containerd runc nvidia-docker2
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 
pip3 install click

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
# THIS IS NECESSARY AS OTHERWISE NVIDIA REPOS TAKE OVER UBUNTU-ONES AND THE VERSIONING GETS BROKEN
sudo rm -rf /etc/apt/sources.list.d/nvidia-*