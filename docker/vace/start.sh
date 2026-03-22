#!/bin/bash

source ./common.sh

#export DISPLAY=:0
#export XAUTHORITY=/home/takuya/.Xauthority

#xhost +local:docker

docker run \
    --gpus all \
    --net=host \
    -itd \
    --shm-size=8G \
    --privileged \
    -v ${GRANDPARENT_DIR}:/workspace \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --workdir /workspace \
    --name $CONTAINER_NAME \
    $IMAGE_NAME

    #-v /tmp/.X11-unix:/tmp/.X11-unix \    
    #--env DISPLAY=$DISPLAY \    