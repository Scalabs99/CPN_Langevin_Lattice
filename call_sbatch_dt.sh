#!/bin/bash

config_path="/hpc/home/nome.cognome/CPN_Langevin_Lattice/simulation_configurations" # path to simulation configuration files folder (without trailing /)
base_config_fname="sample_configuration_lattice.toml" # file di configurazione base
checkpt_path="/hpc/home/nome.cognome/CPN_Langevin_Lattice/checkpoint" # path to simulation checkpoint files folder (without trailing /)

source configuration.conf

# Definisci qui l'array dei diversi valori di dt che vuoi testare
dt_values=(0.001 0.0005 0.00025 0.0001)

for dt_val in "${dt_values[@]}"; do
    # Crea un nome univoco per il file di configurazione specifico di questo dt
    config_fname="config_dt_${dt_val}.toml"
    dest_config="$config_path/$config_fname"

    # Copia il file base e sostituisce (o appende) il nuovo valore di dt e i nomi dei file di output
    sed "s/^dt[[:space:]]*=.*/dt                     = $dt_val/" "$config_path/$base_config_fname" > "$dest_config"
    
    # Personalizza anche i nomi dei file di output e checkpoint nel file .toml temporaneo 
    # per evitare che i job scrivano sugli stessi file
    sed -i "s|energy_filename[[:space:]]*=.*|energy_filename        = \"energies/energy_dt_${dt_val}.txt\"|" "$dest_config"
    sed -i "s|checkpoint_filename[[:space:]]*=.*|checkpoint_filename    = \"checkpoint/checkpoint_dt_${dt_val}.jld\"|" "$dest_config"

    # Nome del job basato anche sul dt
    name=$(date +%Y-%m-%d_%H:%M)"_dt_"_$dt_val

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

    echo "Lancio il job per dt = $dt_val..."
    sbatch ${SBATCH_OPTS[@]} job_sbatch.bash start "$dest_config"
    
    # Un piccolissimo sleep per evitare di sovraccaricare il demone di SLURM
    sleep 1
done

echo "Tutti i job per l'estrapolazione in dt sono stati sottomessi con successo!"
