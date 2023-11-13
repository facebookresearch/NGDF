#!/bin/bash

# Get current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Source ngdf conda env
source activate ngdf

# Source ndf_robot
cd $DIR/ndf_robot && source ndf_env.sh
cd -

