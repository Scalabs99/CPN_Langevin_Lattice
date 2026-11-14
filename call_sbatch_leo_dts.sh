#!/bin/bash

config_path="/leonardo/home/userexternal/dscalabr/CPN_Langevin_Lattice"
base_config_fname="sample_configuration_lattice.toml"
checkpt_path="/hpc/home/nome.cognome/CPN_Langevin_Lattice/checkpoint"

source configuration.conf

# Assicurati che le cartelle per i log e le configurazioni generate esistano
mkdir -p logs
mkdir -p "$config_path/generated_configs"

# Definisci qui la lista dei valori di dt che vuoi simulare
DT_VALUES=("0.001" "0.002" "0.005" "0.007" "0.010")

for dt in "${DT_VALUES[@]}"; do
    # Nome del nuovo file .toml specifico per questo dt
    new_config_fname="config_dt_${dt}.toml"
    new_config_fullpath="$config_path/generated_configs/$new_config_fname"

    # Crea il nuovo .toml sostituendo dt e differenziando i file di output
    sed -e "s/^\(dt\s*=\s*\)[0-9.eE+-]*/\1${dt}/" \
        -e "s|energies/energy.txt|energies/energy_dt_${dt}.txt|" \
        -e "s|checkpoint/checkpoint.jld|checkpoint/checkpoint_dt_${dt}.jld|" \
        -e "s|file.mat|file_dt_${dt}.mat|" \
        "$config_path/$base_config_fname" > "$new_config_fullpath"

    # Nome del job su SLURM
    name=$(date +%Y-%m-%d_%H:%M)"_dt_${dt}"

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

    echo "Sottomissione job per dt = $dt con config: $new_config_fname"
    sbatch "${SBATCH_OPTS[@]}" job_sbatch.bash start "$new_config_fullpath"
done
