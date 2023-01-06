FROM ros-base:v1.0

ENV DEBIAN_FRONTEND noninteractive

ARG USERNAME=ros
USER $USERNAME
RUN mkdir -p /home/ros/phase4ai/ws/arm/src

USER root

ARG USERNAME=ros
USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo apt update && \
    sudo apt install -y --no-install-recommends --allow-unauthenticated \
        software-properties-common curl apache2-utils \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils curl ca-certificates gnupg patch \
        mesa-utils libgl1-mesa-dri ffmpeg tmux libusb* \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
        xvfb x11vnc libcanberra-gtk-module libcanberra-gtk3-module meshlab psmisc python-dev python3-dev \
        vim-tiny sudo firefox-geckodriver ttf-ubuntu-font-family ttf-wqy-zenhei tmuxinator python-dev \
        usbutils joystick cheese libglvnd-dev python3-pip && \
    pip install hidapi && \
    pip install --upgrade setuptools && \
    sudo add-apt-repository -r ppa:fcwu-tw/apps && \
    sudo mkdir /usr/local/ffmpeg && \
    sudo apt-get autoclean

RUN cd /home/ros/ &&\
    git clone --branch v2.50.0 https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    ./scripts/setup_udev_rules.sh && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=release -DFORCE_RSUSB_BACKEND=false -DBUILD_WITH_CUDA=false && make -j$(($(nproc)-1))

USER root

RUN cd /home/ros/librealsense/build && make install
RUN usermod  -a -G video ros

USER $USERNAME

RUN sudo ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg  && \
    sudo chmod -R o+rw /usr/local && \
    pip install empty catkin_pkg lark PyYAML

RUN wget -O arduino-ide.zip https://downloads.arduino.cc/arduino-ide/arduino-ide_2.0.0-rc3_Linux_64bit.zip?_gl=1*1lij0u2*_ga*MTg0MTE3MTE1NS4xNjQwNDI0NjE3*_ga_NEXN8H46L5*MTY0MDQyNDYxNi4xLjEuMTY0MDQyNDY2MC4w && \
    unzip arduino-ide.zip && \
    rm arduino-ide.zip && \
    wget -O arduino-cli.tar.gz https://github.com/arduino/arduino-cli/releases/download/0.20.2/arduino-cli_0.20.2_Linux_64bit.tar.gz && \
    tar -xvf arduino-cli.tar.gz && \
    rm arduino-cli.tar.gz && \
    mkdir -p .local/bin && \
    mv arduino-cli .local/bin
COPY arduino-cli.yaml $HOME/ 
RUN .local/bin/arduino-cli core update-index --config-file arduino-cli.yaml && \
    .local/bin/arduino-cli core update-index && \
    .local/bin/arduino-cli board listall && \
    .local/bin/arduino-cli core install esp32:esp32


ENV HOME=/home/ros \
    SHELL=/bin/bash \
    CUDA_HOME=/usr/local/cuda

RUN mkdir -p /home/ros/.config/lxterminal/
COPY lxterminal.conf /home/ros/.config/lxterminal/lxterminal.conf

RUN python3 -m pip install --upgrade pip && \
    sudo apt update && \
    sudo apt install -y python3-send2trash && \
    sudo apt install -y --no-install-recommends iputils-ping ros-noetic-rviz-visual-tools ros-noetic-ddynamic-reconfigure* && \
    sudo apt install -y ros-noetic-geometric-shapes ros-noetic-moveit-msgs \
        ros-noetic-srdfdom ros-noetic-pybind11-catkin ros-noetic-moveit-resources-* \
        ros-noetic-ompl ros-noetic-warehouse-ros-* ros-noetic-eigenpy \
        ros-noetic-moveit-visual-tools ros-noetic-rosparam-shortcuts \
        ros-noetic-collada-urdf ros-noetic-rospy-message-converter && \
    sudo python3 -m pip install --ignore-installed setuptools testresources && \
    sudo chown -R ros:ros /usr/local && \
    sudo apt-get autoclean

RUN cd /home/ros/librealsense && \
    sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ && \
    sudo /lib/systemd/systemd-udevd --daemon && \
    sudo udevadm control --reload-rules && udevadm trigger

RUN cd /home/ros && \
    git clone https://github.com/crigroup/openrave-installation && \
    cd openrave-installation && \
    ./install-dependencies.sh && \
    ./install-osg.sh && \
    ./install-fcl.sh && \
    ./install-openrave.sh && \
    sudo apt-get autoclean

RUN pip install catkin_pkg && \
    pip install lxml && \
    sudo chmod -R o+rw /usr/local && \
    pip3 uninstall -y sklearn scipy && \
    pip3 install -U sklearn scipy pyrealsense2 configargparse gdown && \
    pip3 install argh natsort diskcache easydict trimesh

RUN sudo apt-get update && \
    sudo apt-get install -y ros-noetic-trac-ik-kinematics-plugin && \
    sudo apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-system-dev \
    libeigen3-dev && \
    sudo apt install -y ros-noetic-tf2* ros-noetic-pcl-* \
    xorg-dev libglu1-mesa-dev python3-dev libsdl2-dev \
    libc++-7-dev libc++abi-7-dev ninja-build libxi-dev gfortran libtbb-dev \
    libosmesa6-dev libudev-dev autoconf libtool flex libkdtree++-dev && \
    sudo apt-get autoclean

RUN sudo DEBIAN_FRONTEND=noninteractive  apt install -y intel-mkl && \
    echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list && \
    sudo apt-get update -y && \
    sudo apt-get install -y nvhpc-22-1 && \
    sudo rm -rf /etc/alternatives/cuda && \
    sudo rm -rf /usr/local/cuda && \
    sudo ln -s /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5  /usr/local/cuda && \
    sudo cp -ans /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/math_libs/11.5/targets/x86_64-linux/include/* /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5/targets/x86_64-linux/include/ && \
    sudo cp -ans /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/math_libs/11.5/targets/x86_64-linux/lib/* /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5/targets/x86_64-linux/lib/ && \
    pip3 uninstall -y tensorflow && sudo chmod -R o+w /usr/bin/ && pip3 install --upgrade numpy && \
    sudo apt-get autoclean

RUN sudo ln /usr/local/bin/pip /usr/bin/pip && \
    pip3 install --upgrade pip && \
    pip3 install lxml && \
    sudo chmod -R o+w /usr/share && \
    pip3 install pyserial && \
    pip3 install returns && \
    pip3 install --upgrade pip && \
    pip3 install torch-scatter pyrender urdfpy pykdtree pybullet scikit-image pytorch-ignite nvisii pytransform3d hydra-core torchmetrics && \
    pip3 install black tabulate gym plotly pytorch_lightning wandb

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN git clone -b v4.1.4 https://github.com/open-mpi/ompi.git && \
    cd ompi && ./autogen.pl && ./configure && make && sudo make install && cd .. && \
    sudo mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include && \
    sudo ln -s  /opt/hpcx/ompi/include  /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi && \
    sudo mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include && \
    sudo ln -s  /opt/hpcx/ompi/include  /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi && \
    sudo mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc && \
    sudo ln -s  /opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include && \
    sudo mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent && \
    sudo ln -s /opt/hpcx/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include && \
    git clone --recursive https://github.com/uber/horovod.git && cd horovod && python3 setup.py clean && python3 setup.py bdist_wheel && pip3 install ./dist/horovod-0.26.1-cp38-cp38-linux_x86_64.whl

RUN pip3 install theseus-ai

RUN sudo apt update && \
    sudo apt install -y python-is-python3 && \
    sudo apt-get autoclean

ARG SSH_PRIVATE_KEY
ARG SSH_PUBLIC_KEY

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh && \
    bash $HOME/miniconda.sh -b -p $HOME/miniconda && \
    rm -rf miniconda.sh

ENV PATH $HOME/miniconda/bin:$PATH

#
# WHATCH OUT! DO READ THIS! SERIOUS SECURITY BREACH CAN HAPPEN OTHERWISE !
#
# To ensure your private SSH keys are not stored in a docker image or snapshot
# you should run the whole sequence of commands in a row.
# Otherwise, the private keys will be stored in a snapshot, and finally
# removed in the next one, but you will leave a trace of a binary which 
# does contain your **GIT PRIVATE KEYS** that can be used by anybody else and compromise you.
# Be careful !
#
RUN conda init && \
    mkdir -p /home/ros/.ssh && \
    chmod 0700 /home/ros/.ssh && \
    ssh-keyscan github.com > /home/ros/.ssh/known_hosts && \
    echo "${SSH_PRIVATE_KEY}" > /home/ros/.ssh/id_rsa && \
    chmod 600 /home/ros/.ssh/id_rsa && \
    echo "${SSH_PUBLIC_KEY}" > /home/ros/.ssh/id_rsa.pub && \
    chmod 600 /home/ros/.ssh/id_rsa.pub && \
    cd $HOME && git clone --recursive git@github.com:facebookresearch/NGDF.git && \
    cd $HOME/NGDF && conda env create -f ngdf_env.yml && conda activate ngdf && pip install -e . && \
    cd $HOME && curl -LO https://github.com/NVIDIA/cub/archive/1.13.1.tar.gz && tar xzf 1.13.1.tar.gz && export CUB_HOME=$PWD/cub-1.13.1 && \
    cd $HOME && git clone -b v0.7.1 https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && PYTORCH3D_NO_NINJA=1 pip install -e . && \
    cd $HOME/NGDF/ndf_robot && pip install -e . && \
    cd $HOME/NGDF/acronym && pip install -e . && \
    cd $HOME/NGDF/ndf_robot && source ndf_env.sh && ./scripts/download_demo_weights.sh && \
    export CUDA_HOME=$HOME/miniconda3/envs/ngdf/pkgs/cuda-toolkit && \
    cd $HOME/NGDF/OMG-Planner/ycb_render && pip install -e . && \
    cd $HOME && git clone https://github.com/eigenteam/eigen-git-mirror && \
    cd $HOME/eigen-git-mirror && mkdir -p build && cd build && cmake .. && sudo make install && \
    cd $HOME/NGDF/OMG-Planner/Sophus && mkdir -p build && cd build && cmake .. && make -j8 && sudo make install && \
    cd $HOME/NGDF/OMG-Planner/layers && pip install -e . && \
    cd $HOME/NGDF/OMG-Planner/orocos_kinematics_dynamics/sip-4.19.3 && python3 configure.py && make && sudo make install && \
    export ROS_PYTHON_VERSION=3 && cd $HOME/NGDF/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl && mkdir -p build && cd build && cmake .. && make -j8 && sudo make install && \
    cd $HOME/NGDF/OMG-Planner/orocos_kinematics_dynamics/python_orocos_kdl && mkdir -p build && cd build && cmake .. && make -j8 && cp PyKDL.so $HOME/miniconda3/envs/ngdf/lib/python3.7/site-packages && \
    cd $HOME/NGDF/differentiable-robot-model && \
    git remote add parent https://github.com/facebookresearch/differentiable-robot-model.git && \
    git fetch parent && pip install -e . && \
    cd $HOME/NGDF/OMG-Planner && pip install -r requirements.txt && pip install -e . && \
    cd $HOME/NGDF/differentiable-robot-model && pip install -e . && \
    cd $HOME/NGDF/contact_graspnet && source compile_pointnet_tfops.sh && cd ../../.. && pip install -e . && \
    cd $HOME/NGDF/theseus && pip install -e . && \
    cd $HOME && git clone --recursive -j8 git://github.com/hjwdzh/Manifold && cd Manifold && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && \
    rm -rf /home/ros/.ssh

ENTRYPOINT ["/usr/bin/lxpanel", "--profile", "LXDE"]
