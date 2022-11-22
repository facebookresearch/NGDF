#!/usr/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$SCRIPT_DIR/../..
DATA_ROOT=$SCRIPT_DIR/../../data

python ngdf/train.py \
    experiment_yml=$SCRIPT_DIR/../../ngdf/config/experiments/perobj_Bottle.yaml \
    data_root=$DATA_ROOT \
    project_root=$PROJECT_ROOT 