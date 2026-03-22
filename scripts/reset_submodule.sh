#!/bin/bash
set -e 

FILE_PATH=$(realpath $0)
DIR_PATH=$(dirname $FILE_PATH)
PARENT_DIR=$(dirname $DIR_PATH)
cd $PARENT_DIR

# ① .gitmodules削除
rm -f .gitmodules

# ② indexからも削除（これが重要）
git rm --cached .gitmodules || true

# ③ modules削除
rm -rf .git/modules

# ④ commit
#git commit -m "remove broken gitmodules"

# rm -rf third_party