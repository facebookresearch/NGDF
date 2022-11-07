#!/usr/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$SCRIPT_DIR/../../..
DATA_ROOT=$SCRIPT_DIR/../../../data

echo "Evaluating NGDF (Intra-category Bottle)..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=False \
    experiment=multobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=NGDF_intraBottle_grasp1k_smth0.01_obs0.1 \
    dataset=acronym_multobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT
