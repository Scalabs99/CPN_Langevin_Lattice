using Serialization
using Random
using CUDA
using MAT
using Dates

function current_time()
    return string(now()) * ": "
end

function open_energy_file(fname::String, mode::String="w", header::Bool=false)
    file = open(fname, mode)
    if header 
        println(file, "# Each line is an energy measure,")
        println(file, "# each column corresponds to the perturbative order: from 0 to the max\n")
    end
    return file
end

function write_line(file::IOStream, array::Vector{T}) where {T}
    for j in 1:size(array)[1]
        @inbounds print(file, array[j], " ")
    end
    println(file)
end

function get_copy_energy_filename(energy_filename, n_copy)
    if n_copy < 2
        return energy_filename
    end
    return energy_filename[1:end-4] * "_copy" * string(n_copy) * ".txt"
end

function get_seconds(time::String)
    seconds = 0
    sep = findfirst('-', time)
    if !(sep === nothing)
        seconds += parse(Int, time[1:sep-1]) * 24 * 60 * 60 
        time = time[sep+1:end]
    end
    i = 0
    for t in reverse(split(time, ':'))
        seconds += parse(Int, t) * 60 ^ i 
        i += 1
    end
    return seconds
end

function get_job_id()
    return get(ENV, "SLURM_JOB_ID", "")
end

function get_job_name()
    return get(ENV, "SLURM_JOB_NAME", "")
end

const SIM_START_TIME = time()

function save_state(fname::String, args)
    cpu_data = Dict(
        "config_fname" => args.config_fname, # Aggiungi questo!
        "iter_start"   => args.iter_start,
        "z"            => Array(args.z),
        "U"            => Array(args.U),
        "ener_meas"    => args.ener_meas === nothing ? nothing : Array(args.ener_meas)
    )
    serialize(fname, cpu_data)
    println(current_time(), "State saved safely to file \"" * fname * "\"")
    return
end

function execute_self(fname::String)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname`
    exec_time = current_time()
    println(exec_time, "Re-executing itself with command: ", command)
    println(" "^length(exec_time), read(command, String))
    return
end

function save_matlab_energy(lat_fname::String, lat)
    matwrite(lat_fname, Dict("energies"=>lat))
    println(current_time(), "Energy history written to file ", lat_fname)
    return
end

function load_C_config(::Type{F}, vac_fname::String, Npoint::I, n_comps::I) where{F <: AbstractFloat, I <: Integer}
    nu_cpu = zeros(F, 2, Npoint, Npoint, n_comps)
    U_cpu  = zeros(F, 2, Npoint, Npoint, 2)

    if vac_fname == ""
        println(current_time(), "[INFO]: Nessun file del vuoto specificato. Inizializzazione al vuoto banale.")
        
        nu_cpu[1, :, :, n_comps] .= F(1)
        U_cpu[1, :, :, :] .= F(1)
        phase = F(2.0 * pi * (n_comps - 1) / n_comps)
        
        U_cpu[1, Npoint, :, 1] .= F(cos(phase))
        U_cpu[2, Npoint, :, 1] .= F(-sin(phase))
        
        return nu_cpu, U_cpu
    end

    if !isfile(vac_fname)
        error("Vacuum file specificato ma non trovato: ", vac_fname)
    end
    
    open(vac_fname, "r") do f
        header_line = readline(f)
        tokens = split(header_line)
        
        stored_N  = parse(Int, tokens[2])
        stored_Lt = parse(Int, tokens[3])
        stored_Lx = parse(Int, tokens[4])
        
        if stored_N != n_comps || stored_Lt != Npoint || stored_Lx != Npoint
            error("Dimensioni incompatibili! File: N=$stored_N, Lt=$stored_Lt, Lx=$stored_Lx")
        end
        
        for x in 1:Npoint
            for t in 1:Npoint
                
                # Legge e inverte i byte (Big-Endian -> Native) per ogni colore
                for c in 1:n_comps
                    nu_cpu[1, t, x, c] = F(ntoh(read(f, Float64)))
                    nu_cpu[2, t, x, c] = F(ntoh(read(f, Float64)))
                end
                
                # Legge e inverte i byte per ogni direzione di gauge
                for mu in 1:2
                    U_cpu[1, t, x, mu] = F(ntoh(read(f, Float64)))
                    U_cpu[2, t, x, mu] = F(ntoh(read(f, Float64)))
                end
                
            end
        end
    end
    return nu_cpu, U_cpu
end

mutable struct LangRK_args{F <: AbstractFloat, I <: Integer}
    const Npoint::I
    const n_meas::I
    const dt::F
    const n_comps::I
    const max_ptord::I
    const measure_every::I
    const n_copies::I
    const cuda_rng::CUDA.RNG
    const en_fname::String
    const config_fname::String
    const checkpt_fname::String
    const max_execution_time::String 
    iter_start::I
    z::CuArray{F, 6} 
    U::CuArray{F, 6} 
    nu_vacuum::CuArray{F, 4} 
    U_vacuum::CuArray{F, 4}  
    const lat_fname::Union{Nothing, String}
    ener_meas::Union{Nothing, CuArray{F, 5}}
end

function get_infos_string(sim_infos::LangRK_args; n_decimals::Int=2, header::String="", sep::String="\n", prepend::String="", append::String="")
    head_spaces = " " ^ length(header)
    msg = header * "Langevin RK Simulation of CP(" * string(sim_infos.n_comps)
    msg *= ") model up to perturbative order " * string(sim_infos.max_ptord) * " (included), on "
    msg *= string(sim_infos.n_copies) * " copies of a "
    msg *= string(sim_infos.Npoint) * "x" * string(sim_infos.Npoint) * " lattice." * sep
    msg *= head_spaces * "Computer-time integration steps of " * string(sim_infos.dt) 
    msg *= ", saving a measurement every " * string(sim_infos.measure_every) * "." * sep
    msg *= head_spaces * "Starting from measurement #" * string(sim_infos.iter_start) * " out of " * string(sim_infos.n_meas)
    if n_decimals <= 0
        percentage = round(Int, sim_infos.iter_start / sim_infos.n_meas * 100)
    else
        percentage = round(sim_infos.iter_start / sim_infos.n_meas * 100 * 10^n_decimals) / 10^n_decimals
    end
    msg *= " (" * string(percentage) * "% of total)."
    return prepend * msg * append
end