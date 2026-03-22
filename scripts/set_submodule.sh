#!/bin/bash

set -e 

FILE_PATH=$(realpath $0)
DIR_PATH=$(dirname $FILE_PATH)
PARENT_DIR=$(dirname $DIR_PATH)
cd $PARENT_DIR

mkdir -p third_party

git submodule add https://github.com/ali-vilab/VACE.git third_party/VACE
git submodule add https://github.com/facebookresearch/sam2.git third_party/sam2
git submodule add https://github.com/facebookresearch/segment-anything.git third_party/segment-anything
git submodule add https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
git submodule add https://github.com/xinyu1205/recognize-anything.git third_party/recognize-anything
git submodule add https://github.com/martin-chobanyan-sdc/RAFT.git third_party/RAFT
git submodule add https://github.com/Wan-Video/Wan2.1 third_party/Wan2.1

git submodule update --init --recursive
git commit -m "add submodules"