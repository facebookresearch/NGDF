# Introduction

This folder contains a mechanism to launch `NGDF` using docker as well as a development 
environment for it, which includes VSCode, ROS, Open3D and nVidia support among others.

## Installation

For the docker machine to work it is necesary to install `docker` and `nVidia docker`.

If you are using `Ubuntu`, you can install the requirements running:
```bash
./bin/install-docker.sh
```

Then, build the docker images running:
```bash
./bin/build.sh
```

Finally, copy the `HOME` folder of the docker machine into a local
folder so that you can develop using the docker machine in a way that changes are persisted.
```bash
./bin/install.sh
```

## Launch

After the installation is sucessful, run:
```bash
./bin/launch.py
```