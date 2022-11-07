#!/usr/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python ngdf/train.py \
    --experiment_yml=perobj_Bottle \
    --data_root=/home/thomasweng/projects/ngdf/data \
    --project_root=/home/thomasweng/projects/ngdf