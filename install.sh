#!/bin/bash

# Pipe all output to log file
exec > >(tee -i /tmp/ngdf_install.log)

# Get directory of current file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Install or update ngdf env
set -e # exit on error
if [ -d "$CONDA_PREFIX/envs/ngdf" ]; then
    conda env update -f ngdf_env.yml
else
    conda env create -f ngdf_env.yml
fi

# # Set up ndf_robot hook
# source activate ngdf
# cd $CONDA_PREFIX
# if [ ! -d "./etc/conda/activate.d" ]; then
#     mkdir -p ./etc/conda/activate.d
#     touch ./etc/conda/activate.d/ndf_env.sh
#     echo "cd $DIR/ndf_robot && source ndf_env.sh && cd -" >> ./etc/conda/activate.d/ndf_env.sh
# fi

# Download ndf_robot pretrained weights
if [ ! -f "$DIR/ndf_robot/src/model_weights/multi_category_weights.pth" ]; then
    cd $DIR/ndf_robot
    source ndf_env.sh
    bash $DIR/ndf_robot/scripts/download_demo_weights.sh
fi