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
    --ckpt_path=data/models/objBottle-archdeepsdf_partial100/default_default_train-graspfields/5_5_3tbhr0ee/checkpoints/epoch=11-step=510732.ckpt \
    --data_path=data/acronym_perobj/grasp-dataset \
    --pc_data_path=data/acronym_perobj/shape-dataset \
    --eval_objs=Bottle

echo "Evaluating Bowl..."
python ngdf/evaluate.py \
    $pb_arg \
    --max_samples=50 \
    --max_iter=3000 \
    --save_images \
    --ckpt_path=data/models/objBowl-archdeepsdf_partial100/default_default_train-graspfields/7_7_3n5713s0/checkpoints/epoch=12-step=550822.ckpt \
    --data_path=data/acronym_perobj/grasp-dataset \
    --pc_data_path=data/acronym_perobj/shape-dataset \
    --eval_objs=Bowl

echo "Evaluating Mug..."
python ngdf/evaluate.py \
    $pb_arg \
    --max_samples=50 \
    --max_iter=3000 \
    --save_images \
    --ckpt_path=data/models/objMug-archdeepsdf_partial100/default_default_train-graspfields/2_2_1rvf96ap/checkpoints/epoch=5-step=253729.ckpt \
    --data_path=data/acronym_perobj/grasp-dataset \
    --pc_data_path=data/acronym_perobj/shape-dataset \
    --eval_objs=Mug