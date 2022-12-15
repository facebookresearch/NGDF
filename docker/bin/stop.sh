#!/bin/bash
container_id=$(docker ps -aqf "name=ros-desktop" | tr -d '\n')
echo "$container_id"
docker stop $container_id

