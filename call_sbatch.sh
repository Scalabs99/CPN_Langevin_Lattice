#!/bin/bash

config_path="/hpc/home/nome.cognome/CPN_Lang_Lattice/simulation_configurations" # path to simulation configuration files folder (without trailing /)
config_fname="sample_configuration.toml" # name of simulation configuration file
checkpt_path="/hpc/home/nome.cognome/CPN_Lang_Lattice/checkpoint" # path to simulation checkpoint files folder (without trailing /)
checkpt_fname="sample_checkpoint.jld" # name of simulation checkpoint file

source configuration.conf

name=$(date +%Y-%m-%d_%H:%M)"_"$config_fname

SBATCH_OPTS=(
    --job-name=$name
    --partition=gpu
    --qos=gpu
    --gres=gpu:a100_80g:1
    --mem=80G
    --time=1-0:00:00
    --output=logs/%x_%j.out
    --error=logs/%x_%j.err
    --mail-user=$NOTIFICATION_EMAIL
    --mail-type=ALL
)

sbatch ${SBATCH_OPTS[@]} job_sbatch.bash start $config_path/$config_fname # to start
# sbatch ${SBATCH_OPTS[@]} job_sbatch.bash load $checkpt_path/$checkpt_fname # to resume
