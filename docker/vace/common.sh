#!/bin/bash

# User Input
IMAGE_NAME=vace

# Container Name
CONTAINER_NAME=${IMAGE_NAME}_${HOSTNAME}

# Get full path
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$CURRENT_DIR")"
GRANDPARENT_DIR="$(dirname "$PARENT_DIR")"

PROJECT_ROOT=${GRANDPARENT_DIR}

# Dockerfile Path
# DOCKERFILE_PATH=${CURRENT_DIR}/Dockerfile
DOCKERFILE_PATH=${CURRENT_DIR}/Dockerfile

# User info of Host
USER_NAME=$(whoami)
USER_ID=$(id -u) #USER_ID=1001
USER_GID=$(id -g)

# Working Dir of Host
WORK_DIR=${PARENT_DIR}

echo "---------- common.sh ------------"
echo "Image Name     = ${IMAGE_NAME}"
echo "Container Name = ${CONTAINER_NAME}"
echo "WORK_DIR       = ${WORK_DIR}"
echo "USER_NAME  = ${USER_NAME}"
echo "USER_ID    = ${USER_ID}"
echo "USER_GID   = ${USER_GID}"
echo "---------------------------------"