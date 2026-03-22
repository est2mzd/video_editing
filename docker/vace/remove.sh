#!/bin/bash

source ./common.sh
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
