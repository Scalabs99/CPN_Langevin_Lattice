#!/bin/bash

if [ $# != 2 ]; then
    echo "This script must be called with exactly 2 arguments."
    echo "It is intended to be run by \"call_sbatch.sh\" or \"resume_sbatch.sh\""
else
    source configuration.conf
    cd $JULIA_FOLDER_PATH/julia
    module load julia
    julia CPN_LangRK_cluster.jl $@
fi
