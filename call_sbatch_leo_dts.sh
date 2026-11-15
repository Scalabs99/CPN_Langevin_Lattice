#!/bin/bash

config_path="/leonardo/home/userexternal/dscalabr/CPN_Langevin_Lattice"
base_config_fname="sample_configuration_lattice.toml"
checkpt_path="/leonardo_scratch/large/userexternal/dscalabr/CPN_PBC/checkpoint"

energies_dir="/leonardo_scratch/large/userexternal/dscalabr/CPN_PBC/energies"
matlab_dir="/leonardo_scratch/large/userexternal/dscalabr/CPN_PBC/energy_site_by_site"

source configuration.conf

N_COMPS="3" # Numero di componenti indipendenti ( quindi sarebbe N-1 )
# Assicurati che le cartelle per i log e le configurazioni generate esistano
mkdir -p logs
mkdir -p "$config_path/generated_configs$N_COMPS"

# Definisci qui la lista dei valori di dt che vuoi simulare
DT_VALUES=("0.001" "0.002" "0.005" "0.007" "0.010")
N_COPIES="5"

for dt in "${DT_VALUES[@]}"; do
    # Nome del nuovo file .toml specifico per questo dt
    new_config_fname="config_CP${N_COMPS}_dt_${dt}.toml"
    new_config_fullpath="$config_path/generated_configs$N_COMPS/$new_config_fname"

    # Crea il nuovo .toml sostituendo dt e differenziando i file di output
    sed -e "s/^\(dt\s*=\s*\)[0-9.eE+-]*/\1${dt}/" \
        -e "s/^\(n_indep_comps\s*=\s*\)[0-9]*/\1${N_COMPS}/" \
        -e "s/^\(n_copies\s*=\s*\)[0-9]*/\1${N_COPIES}/" \
        -e "s|^\(energy_filename\s*=\s*\)\"[^\"]*\"|\1\"${energies_dir}/energy_CP${N_COMPS}_dt_${dt}.txt\"|" \
        -e "s|^\(checkpoint_filename\s*=\s*\)\"[^\"]*\"|\1\"${checkpt_path}/checkpoint_CP${N_COMPS}_dt_${dt}.jld\"|" \
        -e "s|^\(energy_site_by_site_matlab\s*=\s*\)\"[^\"]*\"|\1\"${matlab_dir}/file_CP${N_COMPS}_dt_${dt}.mat\"|" \
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

    checkpt_fname="checkpoint_CP${N_COMPS}_dt_${dt}.jld"

    if [ -f "$checkpt_path/$checkpt_fname" ]; then
        echo "Trovato checkpoint per dt = $dt, sottomissione in modalità RESUME."
        # Passiamo 'resume' e il percorso COMPLETO del checkpoint
        sbatch "${SBATCH_OPTS[@]}" job_sbatch.bash resume "$checkpt_path/$checkpt_fname"
    else
        echo "Nessun checkpoint per dt = $dt, sottomissione da zero (START)."
        # Passiamo 'start' e il percorso COMPLETO della configurazione
        sbatch "${SBATCH_OPTS[@]}" job_sbatch.bash start "$new_config_fullpath"
    fi
done
done
