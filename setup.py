# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

from setuptools import find_packages, setup

requirements = []

__version__ = "0.0.1"

setup(
    name="ngdf",
    version=__version__,
    author="Thomas Weng",
    packages=find_packages(),
    install_requires=requirements,
)
