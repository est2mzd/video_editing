#!/bin/bash

source ./common.sh

docker build \
    --progress=plain \
    -f $DOCKERFILE_PATH \
    --build-arg USER_NAME=${USER_NAME} \
    --build-arg USER_ID=${USER_ID} \
    --build-arg GROUP_ID=${USER_GID} \
    -t $IMAGE_NAME \
    . # "$PROJECT_ROOT"

# --no-cache \
# docker build --no-cache --progress=plain .
