#!/bin/bash

config_path="/leonardo/home/userexternal/dscalabr/CPN_Langevin_Lattice" # path to simulation configuration files folder (without trailing /)
config_fname="sample_configuration_lattice.toml" # name of simulation configuration file
checkpt_path="/hpc/home/nome.cognome/CPN_Langevin_Lattice/checkpoint" # path to simulation checkpoint files folder (without trailing /)
checkpt_fname="sample_checkpoint.jld" # name of simulation checkpoint file

source configuration.conf

name=$(date +%Y-%m-%d_%H:%M)"_"$config_fname

SBATCH_OPTS=(
      	 --job-name=$name
         --ntasks-per-node=1
         --cpus-per-task=1
         --gres=gpu:1
         --partition=boost_usr_prod
         --mem=80G
         --account=INF26_qcdlat
         --time=1-0:00:00
         --output=logs/%x_%j.out
         --error=logs/%x_%j.err
      	 --mail-user=$NOTIFICATION_EMAIL
	 --mail-type=ALL
)

sbatch ${SBATCH_OPTS[@]} job_sbatch.bash start $config_path/$config_fname # to start
# sbatch ${SBATCH_OPTS[@]} job_sbatch.bash load $checkpt_path/$checkpt_fname # to resume
