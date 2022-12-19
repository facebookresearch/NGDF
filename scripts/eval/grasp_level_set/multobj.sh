#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

while getopts p option
do
    case "${option}" in
        p) pb_arg="--eval_pybullet";;
    esac
done

echo "Evaluating Bottle..."
python ngdf/evaluate.py \
    $pb_arg \
    --max_samples=50 \
    --max_iter=3000 \
    --save_images \
    --ckpt_path=data/models/Bottle_intracategory_partial/default_default_train-graspfields/0_0_248dyj7f/checkpoints/epoch=0-step=199999.ckpt \
    --data_path=data/acronym_multobj/grasp-dataset \
    --pc_data_path=data/acronym_multobj/shape-dataset \
    --eval_objs=Bottle