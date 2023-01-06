#!/usr/bin/env python3
import sys
import os 
import click
import subprocess

@click.command()
@click.option("--data-dir", default=lambda: f"{os.environ.get('HOME', '')}/docker-dir", help="Data directory")
@click.option("--desktop", default=False, help="Desktop mode", type=bool)
@click.option("--gpu", default=True, help="GPU Support", type=bool)
def main(data_dir, desktop, gpu):
  entrypoint = "" if desktop else "--entrypoint /bin/bash"
  gpu_support = "-e NVIDIA_DRIVER_CAPABILITIES=all --gpus all \\" if gpu else "\\"
  commands = [
  "xhost +",
  "export LIBGL_ALWAYS_INDIRECT=1",
  f"""
  docker container run --rm -it \
    --cap-add=SYS_PTRACE \
    --security-opt=seccomp:unconfined \
    --security-opt=apparmor:unconfined \
    --privileged {gpu_support}
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --volume=/dev:/dev \
    --device=/dev:/dev \
    --user $(id -u) {entrypoint} \
    --workdir /home/ros \
    --mount type=bind,source="{data_dir}/home/ros",target=/home/ros \
    --name ros-desktop-nvidia \
    --security-opt apparmor:unconfined \
    --net=host \
    --env="DISPLAY" \
    --volume="$HOME/.Xauthority:/home/ros/.Xauthority:rw" \
    ros-desktop:v1.0
    """
    ]
  for command in commands:
    results = subprocess.run(command, shell=True, text=True)

if __name__ == "__main__":
  main()
