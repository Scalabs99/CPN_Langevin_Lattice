using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs_lattice.jl")
include("CPN_kernels_lattice.jl")
include("load_config_lattice.jl")

function initialize_M1!(M1_ker::Array{CompiledKernel})
    # Calcola una volta per tutte le matrici di twist 
    CUDA.@sync begin
        for i in eachindex(M1_ker)
            @inbounds run_kernel(M1_ker[i])
        end
    end 
    return nothing 

end 

function compute_gradients!(
    grad_ker_z::Array{CompiledKernel}, 
    grad_ker_U::Array{CompiledKernel}, 
    zscaled2_ker::Array{CompiledKernel},
    roots_ker::Array{CompiledKernel},
)
    # Questa funzione calcola -grad_U e -grad_z
    CUDA.@sync begin  # Calcola z riscalato per il vuoto e poi lo eleva alla seconda 
        for i in eachindex(zscaled2_ker)  
            @inbounds run_kernel(zscaled2_ker[i])
        end
    end
    CUDA.@sync begin # Calcola la radice e l'inverso della radice 
        for i in eachindex(roots_ker)
            @inbounds run_kernel(roots_ker[i])
        end 
    end
    CUDA.@sync begin 
        for i in eachindex(grad_ker_z)
            @inbounds run_kernel(grad_ker_z[i])
        end 
    end 
    CUDA.@sync begin 
        for i in eachindex(grad_ker_U)
            @inbounds run_kernel(grad_ker_U[i])
        end 
    end 
    return nothing 
end

function compute_exponential!(ExpU_ker::Array{CompiledKernel})

    # Funzione che calcola (ExpiW)*U
    CUDA.@sync begin 
        for i in eachindex(ExpU_ker)
            @inbounds run_kernel(ExpU_ker[i])
        end 
    end 
    return nothing 
end

function Euler_step_z!(
    z::CuArray{F, 6}, 
    z_Eu::CuArray{F, 6}, 
    noise_z::CuArray{F, 6}, 
    gradz::CuArray{F, 6},
    dt::F,
    N_colors::I
   
) where {F <: AbstractFloat, I <:Integer}
    

    @views z_Eu .= z
    CUDA.@sync z_Eu .+= (N_colors * dt .* gradz) .+ (sqrt(F(2) * dt) .* noise_z) 
    @views z_Eu[:, 1, :, :, :, :] .= z[:, 1, :, :, :, :] #Cancella il rumore di macchina 

    return nothing 
end 

function Euler_step_U!(
    U_Eu::CuArray{F, 6}, 
    U::CuArray{F, 6}, 
    grad_U::CuArray{F, 5}, 
    noise_X::CuArray{F, 5}, 
    dt::F, 
    X::CuArray{F, 5}, 
    zero_modo_X::CuArray{F, 3}, 
    ExpU_Eu_ker::Array{CompiledKernel},
    zero_modeX_ker::Array{CompiledKernel}, 
    N_colors::I, 
    Npoint2::I
) where {F <: AbstractFloat, I <: Integer}

    @views U_Eu .= U

    # Costruisci l'incremento X (Forza + Rumore)
    @views X[1, :, :, :, :] .= zero(F) # BLINDA IL BACKGROUND: nessuna fase aggiuntiva!
    
    # Aggiorna solo le fluttuazioni -> il rumore l'ho già messo a zero in tutte le componenti fuorchè al primo ordine 
    CUDA.@sync @views X[2:end, :, :, :, :] .= ((F(N_colors) * dt * F(2)) .* grad_U[2:end, :, :, :, :]) .+ (sqrt(F(2) * dt) .* noise_X[2:end, :, :, :, :])

    # Sottrazione zero mode (usando dims=(2,3) in un solo passaggio)
    CUDA.@sync @inbounds @views zero_modo_X .= CUDA.sum(X, dims=(2,3))[:,1,1,:,:] ./ Npoint2
    
    # Lancia il kernel per togliere lo zero mode da X 
    CUDA.@sync begin 
        for i in eachindex(zero_modeX_ker)
            @inbounds run_kernel(zero_modeX_ker[i])
        end 
    end

    # Lancia il kernel per la moltiplicazione sul gruppo Exp(iX) * U = U_eu
    compute_exponential!(ExpU_Eu_ker)

    # Cancella il rumore di macchina dal background del link aggiornato
    @views U_Eu[:, 1, :, :, :, :] .= U[:, 1, :, :, :, :] 
    
    return nothing
end

function RK_evolution_z!(
    z::CuArray{F, 6}, 
    noise_z::CuArray{F, 6}, 
    gradz_Eu::CuArray{F, 6},
    dt::F,
    grad_z::CuArray{F, 6},
    N_colors::I
) where {F <: AbstractFloat, I <: Integer}
    # esegue l'evoluzione di uno step secondo l'algoritmo RK del secondo ordine
    # per l'equazione di Langevin 
    
    CUDA.@sync @views z[:,2:end,:,:,:,:] .+= (F(N_colors) * F(0.5) * dt) .* (grad_z[:,2:end,:,:,:,:] .+ gradz_Eu[:,2:end,:,:,:,:]) .+ (sqrt(F(2) * dt) .* noise_z[:,2:end,:,:,:,:])
    
    return nothing
end

function RK_evolution_U!(
    U::CuArray{F, 6}, 
    gradU_Eu::CuArray{F, 5}, 
    gradU::CuArray{F, 5}, 
    noise_X::CuArray{F, 5}, 
    dt::F, 
    X::CuArray{F, 5}, 
    zero_modo_X::CuArray{F, 3}, 
    ExpU_ker::Array{CompiledKernel},
    zero_modeX_ker::Array{CompiledKernel},
    N_colors::I, 
    Npoint2::I

) where {F <: AbstractFloat, I <: Integer}

    # Costruisci l'incremento X (Forza + Rumore)
    @views X[1, :, :, :, :] .= zero(F) # BLINDA IL BACKGROUND: nessuna fase aggiuntiva!

    # Aggiorna solo le fluttuazioni -> il rumore l'ho già messo a zero in tutte le componenti fuorchè al primo ordine 
    CUDA.@sync @views X[2:end,:,:,:,:] .= (F(N_colors) * F(2) * F(0.5) * dt) .* (gradU[2:end,:,:,:,:] .+ gradU_Eu[2:end,:,:,:,:]) .+ (sqrt(F(2) * dt) .* noise_X[2:end,:,:,:,:])

    # Sottrazione zero mode (usando dims=(2,3) in un solo passaggio)
    CUDA.@sync @inbounds @views zero_modo_X .= CUDA.sum(X, dims=(2,3))[:,1,1,:,:] ./ Npoint2
    
    # Lancia il kernel per togliere lo zero mode da X 
    CUDA.@sync begin 
        for i in eachindex(zero_modeX_ker)
            @inbounds run_kernel(zero_modeX_ker[i])
        end 
    end

    # Lancia il kernel per la moltiplicazione sul gruppo Exp(iX) 
    compute_exponential!(ExpU_ker)
    
    return nothing 
end 

function compute_energy!(
    energy::CuArray{F, 2},
    ener::CuArray{F, 4},
    Npoint2::I,
    zscaled2_ker::Array{CompiledKernel},
    roots_ker::Array{CompiledKernel},
    compute_ener_ker::Array{CompiledKernel},
    ) where {F <: AbstractFloat, I <: Integer}
    # computes the mean of the energy of all sites at time t


    # Non sottraggo più lo zero modo, l'ho già sottratto negli Euler Step!! 

    CUDA.@sync begin
        for i in eachindex(zscaled2_ker)
            @inbounds run_kernel(zscaled2_ker[i])
        end 
    end 

    CUDA.@sync begin
        for i in eachindex(roots_ker)
            @inbounds run_kernel(roots_ker[i])
        end 
    end 

    CUDA.@sync begin
        for i in eachindex(compute_ener_ker)
            @inbounds run_kernel(compute_ener_ker[i])
        end
    end 
    
    CUDA.@sync @inbounds @views energy.= CUDA.sum(ener, dims=(2, 3))[:,1,1,:] ./ (Npoint2)
    return nothing
end

@inline function reset_noise_z!(noise_z::CuArray{F, 6}, rng::CUDA.RNG) where {F <: AbstractFloat}
    CUDA.fill!(noise_z, zero(F))
    @inbounds @views CUDA.randn!(rng, noise_z[:,2,:,:,:,:])
    return nothing
end

@inline function reset_noise_X!(noise_X::CuArray{F, 5}, rng::CUDA.RNG) where {F <: AbstractFloat}
    CUDA.fill!(noise_X, zero(F))
    @inbounds @views CUDA.randn!(rng, noise_X[2,:,:,:,:]) # X è reale!!! 
    return nothing
end

function main_Lang(args::LangRK_args{F, I, I2}) where {F <: AbstractFloat, I <: Integer, I2 <: Integer}
    
    # Print some logging information
    println(get_infos_string(args; header="[Simulation Infos]: ", prepend="\n", append="\n"))
    println(current_time(), "Variables initialization started.")

    # Extract (almost) all the arguments
    Npoint        = args.Npoint
    n_meas        = args.n_meas
    dt            = args.dt
    n_comps       = args.n_comps
    n_copies      = args.n_copies
    max_ptord     = args.max_ptord
    measure_every = args.measure_every
    cuda_rng      = args.cuda_rng
    en_fname      = args.en_fname
    checkpt_fname = args.checkpt_fname
    z             = args.z 
    U             = args.U 
    nu_vacuum     = args.nu_vacuum
    U_vacuum      = args.U_vacuum

    # Determine if we should save lattice data
    save_lattice = args.lat_fname !== nothing
    if save_lattice
        lat_fname::String = args.lat_fname
        ener_meas::CuArray{F, 5} = args.ener_meas
    end
    
    n_ords = max_ptord + one(max_ptord)
    N_colors = n_comps + one(n_comps)
    Npoint2 = Npoint^2

    # Initialize additional variables 
    z_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    U_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_copies)
    X = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    gradient_z = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    gradz_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    gradient_U = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    gradU_Eu = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    Exp_buffer = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_copies)
    zscaled2 = CuArray{F}(undef, n_ords, Npoint, Npoint, n_copies)
    root_z2 = CuArray{F}(undef, n_ords, Npoint, Npoint, n_copies)
    invroot_z2 = CuArray{F}(undef, n_ords, Npoint, Npoint, n_copies)
    noise_z = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    noise_X = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    M1 = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), N_colors, n_copies)

    energia = CUDA.zeros(F, n_ords, n_copies)
    ener = CUDA.zeros(F, n_ords, Npoint, Npoint, n_copies)
    zero_modo_z = CUDA.zeros(F, I(2), n_ords, n_comps, n_copies)
    zero_modo_X = CUDA.zeros(F, n_ords, I(2), n_copies)

    M1_ker = Array{CompiledKernel}(undef, n_copies)
    zscaled2_ker = Array{CompiledKernel}(undef, n_copies); 
    roots_ker = Array{CompiledKernel}(undef, n_copies); 
    zscaled2_ker_Eu = Array{CompiledKernel}(undef, n_copies);
    roots_ker_Eu = Array{CompiledKernel}(undef, n_copies);
    gradker_z = Array{CompiledKernel}(undef, n_copies); 
    gradker_z_Eu = Array{CompiledKernel}(undef, n_copies); 
    gradker_U = Array{CompiledKernel}(undef, n_copies); 
    gradker_U_Eu = Array{CompiledKernel}(undef, n_copies); 
    ExpU_ker = Array{CompiledKernel}(undef, n_copies); 
    ExpU_Eu_ker = Array{CompiledKernel}(undef, n_copies); 
    zero_mode_z_ker = Array{CompiledKernel}(undef, n_copies); 
    zero_mode_X_ker = Array{CompiledKernel}(undef, n_copies); 
    compute_ener_ker = Array{CompiledKernel}(undef, n_copies); 

    en_fnames = [en_fname]
    for i in 2:n_copies
        push!(en_fnames, get_copy_energy_filename(en_fname, i))
    end

    # Handle energy measurement array and file opening
    if args.iter_start == 1
        mean_energy_files = [open_energy_file(en_fname, "w", true)] # with header
        for i in 2:n_copies
            push!(mean_energy_files, open_energy_file(en_fnames[i], "w", true))
        end
    else
        mean_energy_files = [open_energy_file(en_fname, "a", false)] # without header, appending
        for i in 2:n_copies
            push!(mean_energy_files, open_energy_file(en_fnames[i], "a", false))
        end
    end
    
    # creazione delle funzioni a parametri fissati
    f_z2, f_grad_z, f_grad_U, f_zeromode_z, f_zeromode_X, f_ener, f_twist, f_roots, f_exp = create_kernels(F, Npoint, max_ptord, n_comps)

    # compilazione dei kernel e creazione di struct con i kernel compilati

    for i in 1:n_copies
        @inbounds M1_ker[i] = compile_kernel(f_twist, (M1[:,:,:,:,:,:,i],), Npoint2)
        @inbounds zscaled2_ker[i] = compile_kernel(f_z2, (z[:,:,:,:,:,i], nu_vacuum, zscaled2[:,:,:,i]), Npoint2)
        @inbounds roots_ker[i] = compile_kernel(f_roots, (zscaled2[:,:,:,i], root_z2[:,:,:,i], invroot_z2[:,:,:,i]), Npoint2)
        @inbounds gradker_z[i] = compile_kernel(f_grad_z, (z[:,:,:,:,:,i], nu_vacuum, U[:,:,:,:,:,i], gradient_z[:,:,:,:,:,i], root_z2[:,:,:,i], invroot_z2[:,:,:,i], M1[:,:,:,:,:,:,i]), Npoint2)
        @inbounds gradker_U[i] = compile_kernel(f_grad_U, (gradient_U[:,:,:,:,i], U[:,:,:,:,:,i], z[:,:,:,:,:,i], M1[:,:,:,:,:,:,i], root_z2[:,:,:,i], nu_vacuum), Npoint2)
        @inbounds ExpU_ker[i] = compile_kernel(f_exp, (X[:,:,:,:,i], Exp_buffer[:,:,:,:,:,i], U[:,:,:,:,:,i], U[:,:,:,:,:,i]), Npoint2)
        
        @inbounds zscaled2_ker_Eu[i] = compile_kernel(f_z2, (z_Eu[:,:,:,:,:,i], nu_vacuum, zscaled2[:,:,:,i]), Npoint2)
        @inbounds roots_ker_Eu[i] = compile_kernel(f_roots, (zscaled2[:,:,:,i], root_z2[:,:,:,i], invroot_z2[:,:,:,i]), Npoint2)
        @inbounds gradker_z_Eu[i] = compile_kernel(f_grad_z, (z_Eu[:,:,:,:,:,i], nu_vacuum, U_Eu[:,:,:,:,:,i], gradz_Eu[:,:,:,:,:,i], root_z2[:,:,:,i], invroot_z2[:,:,:,i], M1[:,:,:,:,:,:,i]), Npoint2)
        @inbounds gradker_U_Eu[i] = compile_kernel(f_grad_U, (gradU_Eu[:,:,:,:,i], U_Eu[:,:,:,:,:,i], z_Eu[:,:,:,:,:,i], M1[:,:,:,:,:,:,i], root_z2[:,:,:,i], nu_vacuum), Npoint2)
        @inbounds ExpU_Eu_ker[i] = compile_kernel(f_exp, (X[:,:,:,:,i], Exp_buffer[:,:,:,:,:,i], U[:,:,:,:,:,i], U_Eu[:,:,:,:,:,i]), Npoint2)
  
        @inbounds zero_mode_z_ker[i] = compile_kernel(f_zeromode_z, (z[:,:,:,:,:,i], zero_modo_z[:,:,:,i]), Npoint2)
        @inbounds zero_mode_X_ker[i] = compile_kernel(f_zeromode_X, (X[:,:,:,:,i], zero_modo_X[:,:,i]), Npoint2)
        @inbounds compute_ener_ker[i] = compile_kernel(f_ener, (z[:,:,:,:,:,i], U[:,:,:,:,:,i], ener[:,:,:,i], root_z2[:,:,:,i], M1[:,:,:,:,:,:,i], nu_vacuum), Npoint2)
    end

    initialize_M1!(M1_ker)
    
    print(current_time(), "Variables initialization and Kernel compilation successfully completed. ")
    println("Simulation is starting...")

    jobid = get_job_id()

    # Main computation loop
    for t = args.iter_start:n_meas
        for _ in 1:measure_every
            # Genera nuovo rumore gaussiano 
            reset_noise_z!(noise_z, cuda_rng)
            reset_noise_X!(noise_X, cuda_rng)
            # --- Tolgo lo Zero - Mode dal rumore;
            # --- Sottrai la media spaziale dal rumore appena generato;
            noise_mean_z = CUDA.sum(noise_z[:,2:2,:,:,:,:], dims=(3, 4)) ./ Npoint2
            noise_mean_X = CUDA.sum(noise_X[2:2,:,:,:,:], dims=(2, 3)) ./ Npoint2
            CUDA.@sync @inbounds @views noise_z[:,2:2,:,:,:,:] .-= noise_mean_z
            CUDA.@sync @inbounds @views noise_X[2:2,:,:,:,:] .-= noise_mean_X

            # Calcola -grad_z e -grad_U per lo step di Eulero dopo aver calcolato le radici
            compute_gradients!(gradker_z, gradker_U, zscaled2_ker, roots_ker)
            # Fai gli step di Eulero per i due campi 
            Euler_step_U!(U_Eu, U, gradient_U, noise_X, dt, X, zero_modo_X, ExpU_Eu_ker, zero_mode_X_ker, N_colors, Npoint2)
            Euler_step_z!(z, z_Eu, noise_z, gradient_z, dt, N_colors)
            
            # Calcola ora i gradienti per lo step RK 
            compute_gradients!(gradker_z_Eu, gradker_U_Eu, zscaled2_ker_Eu, roots_ker_Eu)
            RK_evolution_U!(U, gradU_Eu, gradient_U, noise_X, dt, X, zero_modo_X, ExpU_ker, zero_mode_X_ker, N_colors, Npoint2)
            RK_evolution_z!(z, noise_z, gradz_Eu, dt, gradient_z , N_colors)
            # Sottrai lo zero mode da z alla fine di ogni singolo step per evitare il drift
            CUDA.@sync @inbounds @views zero_modo_z[:,2:end,:,:] .= CUDA.sum(z[:,2:end,:,:,:,:], dims=(3, 4))[:,:,1,1,:,:] ./ Npoint2
            CUDA.@sync begin 
                for i in eachindex(zero_mode_z_ker)
                    @inbounds run_kernel(zero_mode_z_ker[i])
                end 
            end
        end
        
        CUDA.@allowscalar begin
            println("Campione rumore X: ", noise_X[2, 1, 1, 1, 1])
            println("Campione campo z: ", z[1, 2, 1, 1, 1, 1])
            println("Campione campo U: ", U[1, 2, 1, 1, 1, 1])
        end
        
        compute_energy!(energia, ener, Npoint2, zscaled2_ker, roots_ker, compute_ener_ker)
           
        
        # Store energy measurements if saving lattice
        if save_lattice
            @inbounds @views ener_meas[:,:,:,:,t] .= ener
        end
        
        for i in 1:n_copies
            @inbounds @views write_line(mean_energy_files[i], Array(energia[:, i]))
        end
        args.iter_start = t + 1
        
        if get_remaining_time(jobid) < args.max_saving_time
            println(current_time(), "Saving status.")
            save_state(checkpt_fname, args)  # Z e U sono salvati automaticamente qui! Non ho più bisogno di x_back  
            println(current_time(), "Saving status completed.")
            execute_self(checkpt_fname)
            return
        end
    end
    
    # Final cleanup and file operations
    for i in 1:n_copies
        close(mean_energy_files[i])
        print(current_time(), "Mean energy ")
        n_copies > 1 && print("of copy $i ")
        println("written to file: ", en_fnames[i])
    end

    # Saving energy site-by-site on matlab file
    save_lattice && save_matlab_energy(lat_fname, Array(ener_meas))
    println(current_time(), "All measurements successfully taken.")
    
    return
end

function launch_main_Lang(config_fname::String)
    # Starts from scratch
    # Checking if configuration file exists and has all the mandatory parameters
    if !isfile(config_fname)
        error("Configuration file not found: ", config_fname)
    end
    check_required_keys(config_fname, true)
    curr_time = current_time()
    println(curr_time, "Configuration file found.")
    curr_time = " "^(length(curr_time)-length("[INFO]: ")) * "[INFO]: "
    println(curr_time, "Configuration file: ", config_fname)
    # Load config file
    conf = parse_config_file(config_fname)
    # checking writeability of checkpoint and energy filenames
    if !is_file_writeable(conf.checkpt_fname)
        error("Checkpoint file ", conf.checkpt_fname, " is not writeable.")
    end
    if !is_file_writeable(conf.en_fname)
        error("Energy file ", conf.en_fname, " is not writeable.")
    end
    # Setting specific rng seeds, if provided
    cuda_rng = conf.cuda_seed == 0 ? CUDA.RNG() : CUDA.RNG(conf.cuda_seed)

    floatType = typeof(conf.dt)
    n_ords = conf.max_ptord + one(conf.max_ptord)
    N_colors = conf.n_comps + one(conf.n_comps)
    
    nu_cpu, U_vac_cpu = load_C_config(floatType, conf.vac_fname, conf.Npoint, N_colors)
    nu_vacuum = CuArray(floatType.(nu_cpu))
    U_vacuum = CuArray(floatType.(U_vac_cpu))
    
    z_init = CUDA.zeros(floatType, 2, n_ords, conf.Npoint, conf.Npoint, conf.n_comps, conf.n_copies)
    U_init = CUDA.zeros(floatType, 2, n_ords, conf.Npoint, conf.Npoint, 2, conf.n_copies)
    
    CUDA.@sync begin
        for c in 1:conf.n_copies
            @views z_init[:, 1, :, :, 1:conf.n_comps, c] .= nu_vacuum[:, :, :, 1:conf.n_comps]
            @views U_init[:, 1, :, :, :, c] .= U_vacuum
        end
    end
    
    args = LangRK_args(
        conf.Npoint, conf.n_meas, conf.dt, conf.n_comps, conf.max_ptord, conf.measure_every, conf.n_copies,
        cuda_rng, conf.en_fname, config_fname, conf.checkpt_fname, conf.max_saving_time, one(conf.Npoint),
        z_init, U_init, nu_vacuum, U_vacuum, 
        conf.lat_file, isnothing(conf.lat_file) ? nothing : CUDA.zeros(floatType, n_ords, conf.Npoint, conf.Npoint, conf.n_copies, conf.n_meas)
    )
    
    println(curr_time, "Checkpoint file: ", args.checkpt_fname)
    println(curr_time, "Energy file: ", args.en_fname)
    println(curr_time, "Starting new simulation...")
    main_Lang(args)

    return
end

function resume_main_Lang(checkpt_fname::String)
    # Resumes execution
    if !isfile(checkpt_fname)
        error("Checkpoint file $checkpt_fname not found.")
        return
    end
    # Load config file
    args = deserialize(checkpt_fname)
    curr_time = current_time()
    println(curr_time, "Checkpoint file found.")
    curr_time = " "^(length(curr_time)-length("[INFO]: ")) * "[INFO]: "
    println(curr_time, "Configuration file: ", args.config_fname, " (could be outdated).")
    println(curr_time, "Checkpoint file: ", checkpt_fname)
    println(curr_time, "Energy file: ", args.en_fname)
    println(curr_time, "Resuming execution...")
    main_Lang(args)
    return
end
