#!/usr/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$SCRIPT_DIR/../../..
DATA_ROOT=$SCRIPT_DIR/../../../data

echo "Evaluating NGDF..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=False\
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=NGDF_grasp1k_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Oracle O1..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=O1_known_fixed_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Oracle OMG..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=OMG_known_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Baseline B1..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=B1_CG_fixed_smth0.01_obs0.1_maxscore \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Baseline B2..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=B2_CG_fixed_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Baseline B3..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=B3_CG_proj_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT

echo "Evaluating Baseline B4..."
python -m OMG-Planner.omg_bullet.panda_scene \
    render=True \
    experiment=perobj_reach_grasp \
    eval=1obj_float_rotpos_nograv \
    variant=B4_CG_omg_smth0.01_obs0.1 \
    dataset=acronym_perobj \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT
