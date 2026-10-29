using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs_lattice.jl")
include("CPN_kernels_lattice.jl")
include("load_config_lattice.jl")

function compute_gradients!(
    grad_ker_z::CompiledKernel, 
    grad_ker_U::CompiledKernel
)
    # Questa funzione calcola -grad_U e -grad_z
    run_kernel(grad_ker_z)
    run_kernel(grad_ker_U)

    return nothing 
end

function compute_exponential!(ExpU_ker::CompiledKernel)

    # Funzione che calcola (ExpiW)*U
   
    run_kernel(ExpU_ker)
       
    return nothing 
end

function Euler_step_z!(
    z::CuArray{F, 6}, 
    z_Eu::CuArray{F, 6}, 
    noise_z::CuArray{F, 6}, 
    gradz::CuArray{F, 6},
    dt::F,
    n_comps::I
   
) where {F <: AbstractFloat, I <:Integer}
    

    z_Eu .= z
    z_Eu .+= (n_comps * dt .* gradz) .+ (sqrt(dt) .* noise_z) # Var(noise) = Var(Re(noise)) + Var(Im(noise)) --> non serve il fattore radice di 2!!!
    @views z_Eu[:, 1, :, :, :, :] .= z[:, 1, :, :, :, :] #Cancella il rumore di macchina 

    return nothing 
end 

function Euler_step_U!(
    U_Eu::CuArray{F, 6}, 
    U::CuArray{F, 6}, 
    grad_U::CuArray{F, 5}, 
    dt::F, 
    X::CuArray{F, 5}, 
    zero_modo_X::CuArray{F, 3},
    noise_X::CuArray{F, 5},
    ExpU_Eu_ker::CompiledKernel,
    zero_X_ker::CompiledKernel,
    n_comps::I, 
    Npoint2::I
) where {F <: AbstractFloat, I <: Integer}

    U_Eu .= U

    # Costruisci l'incremento X (Forza + Rumore)
    @inbounds @views X[1, :, :, :, :] .= zero(F) # BLINDA IL BACKGROUND: nessuna fase aggiuntiva!
    
    # Aggiorna solo le fluttuazioni -> il rumore l'ho già messo a zero in tutte le componenti fuorchè al primo ordine 
    @inbounds @views X[2:end, :, :, :, :] .= ((F(n_comps) * dt * F(2)) .* grad_U[2:end, :, :, :, :]) .+ (sqrt(F(2) * dt) .* noise_X[2:end, :, :, :, :])
 
    # Sottrazione zero mode (usando dims=(2,3) in un solo passaggio)
    CUDA.@sync @inbounds @views zero_modo_X .= CUDA.sum(X, dims=(2,3))[:,1,1,:,:] ./ Npoint2

    
    # Lancia il kernel per togliere lo zero mode da X 
    #run_kernel(zero_X_ker) 

    # Lancia il kernel per la moltiplicazione sul gruppo Exp(iX) * U = U_eu
    compute_exponential!(ExpU_Eu_ker)

    # Cancella il rumore di macchina dal background del link aggiornato
    @inbounds @views U_Eu[:, 1, :, :, :, :] .= U[:, 1, :, :, :, :] 
    
    return nothing
end

function RK_evolution_z!(
    z::CuArray{F, 6}, 
    noise_z::CuArray{F, 6}, 
    gradz_Eu::CuArray{F, 6},
    dt::F,
    grad_z::CuArray{F, 6},
    n_comps::I
) where {F <: AbstractFloat, I <: Integer}
    # esegue l'evoluzione di uno step secondo l'algoritmo RK del secondo ordine
    # per l'equazione di Langevin 
    
    @inbounds @views z[:,2:end,:,:,:,:] .+= (F(n_comps) * F(0.5) * dt) .* (grad_z[:,2:end,:,:,:,:] .+ gradz_Eu[:,2:end,:,:,:,:]) .+ (sqrt(dt) .* noise_z[:,2:end,:,:,:,:])
    
    return nothing
end

function RK_evolution_U!(
    U::CuArray{F, 6}, 
    U_temp::CuArray{F, 6},
    gradU_Eu::CuArray{F, 5}, 
    gradU::CuArray{F, 5}, 
    dt::F, 
    X::CuArray{F, 5}, 
    zero_modo_X::CuArray{F, 3}, 
    noise_X::CuArray{F, 5},
    ExpU_ker::CompiledKernel,
    zero_X_ker::CompiledKernel,
    n_comps::I,
    Npoint2::I

) where {F <: AbstractFloat, I <: Integer}

    # Costruisci l'incremento X (Forza + Rumore)
    @inbounds @views X[1, :, :, :, :] .= zero(F) # BLINDA IL BACKGROUND: nessuna fase aggiuntiva!

    # Aggiorna solo le fluttuazioni -> il rumore l'ho già messo a zero in tutte le componenti fuorchè al primo ordine 
    @inbounds @views X[2:end,:,:,:,:] .= (F(n_comps) * F(2) * F(0.5) * dt) .* (gradU[2:end,:,:,:,:] .+ gradU_Eu[2:end,:,:,:,:]) .+ (sqrt(F(2) * dt) .* noise_X[2:end,:,:,:,:])

    # Sottrazione zero mode (usando dims=(2,3) in un solo passaggio)
    CUDA.@sync @inbounds @views zero_modo_X .= CUDA.sum(X, dims=(2,3))[:,1,1,:,:] ./ Npoint2
    
    # Lancia il kernel per togliere lo zero mode da X 
    run_kernel(zero_X_ker)
     

    # Lancia il kernel per la moltiplicazione sul gruppo Exp(iX) 
    compute_exponential!(ExpU_ker)

    @inbounds @views U[:,2:end,:,:,:,:] .= U_temp[:,2:end,:,:,:,:]
    
    return nothing 
end 

function compute_energy!(
    energy::CuArray{F, 2},
    ener::CuArray{F, 4},
    Npoint2::I,
    compute_ener_ker::CompiledKernel,
    ) where {F <: AbstractFloat, I <: Integer}
    # computes the mean of the energy of all sites at time t


    # Non sottraggo più lo zero modo, l'ho già sottratto negli Euler Step!! 
    run_kernel(compute_ener_ker)
    
    
    CUDA.@sync @inbounds @views energy.= CUDA.sum(ener, dims=(2, 3))[:,1,1,:] ./ (Npoint2) #Questa è la vera energia. Il kernel calcola ogni termine della somma
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


function main_Lang(args::LangRK_args{F, I}) where {F <: AbstractFloat, I <: Integer}
    
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
    

    # Determine if we should save lattice data
    save_lattice = args.lat_fname !== nothing
    if save_lattice
        lat_fname::String = args.lat_fname
        ener_meas::CuArray{F, 5} = args.ener_meas
    end

    z_fname::String = "z_mat_file.mat"
    
    n_ords = max_ptord + one(max_ptord)
    Npoint2 = Npoint^2

    # Buffers used in computations (for the kernels)
    vec_buffer_comp = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    sc_buffA        = CuArray{F}(undef, n_ords, Npoint, Npoint, n_copies)
    sc_buffB        = CuArray{F}(undef, n_ords, Npoint, Npoint, n_copies)
    c_sc_buffA      = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_copies)
    c_sc_buffB      = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_copies)

    # Initialize additional variables 
    z_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    U_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_copies)
    U_temp = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_copies)
    X = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    gradient_z = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    gradz_Eu = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    gradient_U = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    gradU_Eu = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    Exp_buffer = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_copies)
    noise_z = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies)
    noise_X = CuArray{F}(undef, n_ords, Npoint, Npoint, I(2), n_copies)
    M1 = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, I(2), n_comps, n_copies)
    zero_modo_z = CUDA.zeros(F, I(2), n_ords, n_comps, n_copies)
    zero_modo_X = CUDA.zeros(F, n_ords, I(2), n_copies) 


    energia = CUDA.zeros(F, n_ords, n_copies)
    ener = CUDA.zeros(F, n_ords, Npoint, Npoint, n_copies)

    z_meas = CuArray{F}(undef, I(2), n_ords, Npoint, Npoint, n_comps, n_copies, n_meas)

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
    f_grad_z, f_grad_U, f_ener, f_twist, f_corr_z, f_proj_noise, f_exp, f_gauge, f_apply_gauge, f_zero_X = create_kernels(F, Npoint, max_ptord, n_comps, n_copies)

    # compilazione dei kernel e creazione di struct con i kernel compilati

    
    M1_ker = compile_kernel(f_twist, (M1,), Npoint2, n_copies)
    gradker_z = compile_kernel(f_grad_z, (z, U, gradient_z, M1, vec_buffer_comp, c_sc_buffA), Npoint2, n_copies)
    gradker_U = compile_kernel(f_grad_U, (gradient_U, U, z, M1, c_sc_buffB, vec_buffer_comp), Npoint2, n_copies)
    ExpU_ker = compile_kernel(f_exp, (X, Exp_buffer, U, U_temp, sc_buffA, sc_buffB), Npoint2, n_copies)
    z_corr_ker = compile_kernel(f_corr_z, (z, sc_buffA, vec_buffer_comp), Npoint2, n_copies)
    gauge_ker = compile_kernel(f_gauge, (z, c_sc_buffA), Npoint2, n_copies)
    apply_gauge_ker = compile_kernel(f_apply_gauge, (z, U, c_sc_buffA, vec_buffer_comp, U_temp), Npoint2, n_copies)
        
    gradker_z_Eu = compile_kernel(f_grad_z, (z_Eu, U_Eu, gradz_Eu, M1, vec_buffer_comp, c_sc_buffA), Npoint2, n_copies)
    gradker_U_Eu = compile_kernel(f_grad_U, (gradU_Eu, U_Eu, z_Eu, M1, c_sc_buffB, vec_buffer_comp), Npoint2, n_copies)
    ExpU_Eu_ker = compile_kernel(f_exp, (X, Exp_buffer, U, U_Eu, sc_buffA, sc_buffB), Npoint2, n_copies)
    zEu_coor_ker = compile_kernel(f_corr_z, (z_Eu, sc_buffA, vec_buffer_comp), Npoint2, n_copies)

    zero_X_ker = compile_kernel(f_zero_X, (X, zero_modo_X), Npoint2, n_copies)
    
    noise_ker = compile_kernel(f_proj_noise, (noise_z, z, c_sc_buffA, vec_buffer_comp), Npoint2, n_copies)
  
    compute_ener_ker = compile_kernel(f_ener, (z, U, ener, M1, c_sc_buffA, vec_buffer_comp), Npoint2, n_copies)

    run_kernel(M1_ker)
    
    print(current_time(), "Variables initialization and Kernel compilation successfully completed. ")
    println("Simulation is starting...")

    CUDA.@allowscalar begin 
        println("Campione campo U all'ordine zero: ", U[1, 1, 1, 1, 1, 1])
        println("z_Re al tempo zero: ", z[1, 1, 1, 1, 1, 1])
        println("z_Im al tempo zero: ", z[2, 1, 1, 1, 1, 1])
    end 


    # Main computation loop
    for t = args.iter_start:n_meas
        for _ in 1:measure_every
            # Genera nuovo rumore gaussiano 
            reset_noise_z!(noise_z, cuda_rng)
            reset_noise_X!(noise_X, cuda_rng)
            # --- Tolgo lo Zero - Mode dal rumore;
            # --- Sottrai la media spaziale dal rumore appena generato;
            #=@inbounds @views noise_mean_z = CUDA.sum(noise_z[:,2:2,:,:,:,:], dims=(3, 4)) ./ Npoint2
            @inbounds @views noise_mean_X = CUDA.sum(noise_X[2:2,:,:,:,:], dims=(2, 3)) ./ Npoint2
            @inbounds @views noise_z[:,2:2,:,:,:,:] .-= noise_mean_z
            @inbounds @views noise_X[2:2,:,:,:,:] .-= noise_mean_X=#
            run_kernel(noise_ker)

            # Calcola -grad_z e -grad_U per lo step di Eulero dopo aver calcolato le radici
            compute_gradients!(gradker_z, gradker_U)

            if t <= 100
                # grad_U ha dimensioni: (n_ords, Npoint, Npoint, mu, copies)
                # Calcoliamo la media per l'ordine 2 (il primo non banale) e mu=1,2 sulla copia 1
                @views media_grad_U_mu1 = CUDA.sum(gradient_U[2, :, :, 1, 1]) / Npoint2
                @views media_grad_U_mu2 = CUDA.sum(gradient_U[2, :, :, 2, 1]) / Npoint2
                @views media_grad_zRe = CUDA.sum(gradient_z[1, 2, :, :, 5, 1]) / Npoint2 
                @views media_grad_zIm = CUDA.sum(gradient_z[2, 2, :, :, 5, 1]) / Npoint2
                println("Step $t | Media grad_U (Ord 2, mu=1): ", media_grad_U_mu1)
                println("Step $t | Media grad_U (Ord 2, mu=2): ", media_grad_U_mu2)
                println("Step $t | Media grad_zRe (Ord 2, n=1): ", media_grad_zRe)
                println("Step $t | Media grad_zIm (Ord 2, n=1): ", media_grad_zIm)
            end

            # Fai gli step di Eulero per i due campi 
            Euler_step_U!(U_Eu, U, gradient_U, dt, X, zero_modo_X, noise_X, ExpU_Eu_ker, zero_X_ker, n_comps, Npoint2)
            Euler_step_z!(z, z_Eu, noise_z, gradient_z, dt, n_comps)
            run_kernel(zEu_coor_ker)
            
            # Calcola ora i gradienti per lo step RK 
            compute_gradients!(gradker_z_Eu, gradker_U_Eu)
            RK_evolution_U!(U, U_temp, gradU_Eu, gradient_U, dt, X, zero_modo_X, noise_X, ExpU_ker, zero_X_ker, n_comps, Npoint2)
            RK_evolution_z!(z, noise_z, gradz_Eu, dt, gradient_z , n_comps)

            # Calcola la media (il modo zero) lungo spazio (dim 3) e tempo (dim 4)
            # Usiamo 1:1 sulla dimensione 5 per preservare la shape per il broadcasting
            CUDA.@sync @inbounds @views zero_modo_z = CUDA.sum(z[:, 2:end, :, :, 1:1, :], dims=(3, 4)) ./ Npoint2

            # Sottrai il modo zero in-place 
            # Avviene per: parte reale e imm (:), ordini > 0 (2:end), su tutto il reticolo (:),
            # SOLO sulla prima componente (1:1), e per tutte le repliche (:)
            CUDA.@sync @inbounds @views z[:, 2:end, :, :, 1:1, :] .-= zero_modo_z
            #run_kernel(z_corr_ker) # Solo alla fine normalizza !! 
            run_kernel(gauge_ker)
            run_kernel(apply_gauge_ker)
            run_kernel(z_corr_ker)
            if t <= 3
                println("[DIAGNOSTICA] Salvataggio del tensore z per lo step $t...")
                # Trasferisce l'array dalla GPU alla CPU con Array(z) e lo salva
                serialize("z_diagnostica_step_$(t).jls", Array(z))
            end


        end
        
        compute_energy!(energia, ener, Npoint2, compute_ener_ker)
           
        
        # Store energy measurements if saving lattice
        if save_lattice
            @inbounds @views ener_meas[:,:,:,:,t] .= ener
        end

        @inbounds @views z_meas[:,:,:,:,:,:,t] .= z
        
        for i in 1:n_copies
            @inbounds @views write_line(mean_energy_files[i], Array(energia[:, i]))
        end
        args.iter_start = t + 1
        
        end_time = SIM_START_TIME + get_seconds(args.max_execution_time)

        println("Iterazione $t completata. Tempo passato dall'avvio: ", round(time() - SIM_START_TIME), " sec.")

        if time() >= end_time
            println("[INFO]: Tempo massimo di esecuzione raggiunto. Avvio salvataggio sicuro.")
            save_state(args.checkpt_fname, args)
            execute_self(args.checkpt_fname)
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
    save_matlab_energy(z_fname, Array(z_meas))
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
    
    nu_cpu, U_vac_cpu = load_C_config(floatType, conf.vac_fname, conf.Npoint, conf.n_comps)
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
        cuda_rng, conf.en_fname, config_fname, conf.checkpt_fname, conf.max_execution_time, one(conf.Npoint),
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
    if !isfile(checkpt_fname)
        error("Checkpoint file $checkpt_fname not found.")
    end
    
    curr_time = current_time()
    println(curr_time, "Checkpoint file found: ", checkpt_fname)
    
    # Carica il dizionario con i dati CPU
    cpu_data = deserialize(checkpt_fname)
    config_fname = cpu_data["config_fname"]
    
    println(curr_time, "Configuration file: ", config_fname)
    println(curr_time, "Resuming execution...")
    
    # 1. Ricarica la configurazione di base partendo dal TOML
    conf = parse_config_file(config_fname)
    cuda_rng = conf.cuda_seed == 0 ? CUDA.RNG() : CUDA.RNG(conf.cuda_seed)
    floatType = typeof(conf.dt)
    n_ords = conf.max_ptord + one(conf.max_ptord)
    
    nu_cpu, U_vac_cpu = load_C_config(floatType, conf.vac_fname, conf.Npoint, conf.n_comps)
    nu_vacuum = CuArray(floatType.(nu_cpu))
    U_vacuum = CuArray(floatType.(U_vac_cpu))
    
    # 2. Crea la struct degli argomenti, ma invece di partire da zero...
    # INIETTA DIRETTAMENTE I DATI GPU RICAVATI DAL CHECKPOINT!
    z_resume = CuArray(cpu_data["z"])
    U_resume = CuArray(cpu_data["U"])
    ener_meas_resume = cpu_data["ener_meas"] === nothing ? nothing : CuArray(cpu_data["ener_meas"])
    
    args = LangRK_args(
        conf.Npoint, conf.n_meas, conf.dt, conf.n_comps, conf.max_ptord, conf.measure_every, conf.n_copies,
        cuda_rng, conf.en_fname, config_fname, conf.checkpt_fname, conf.max_execution_time, 
        cpu_data["iter_start"], # L'iterazione salvata
        z_resume, U_resume, nu_vacuum, U_vacuum, 
        conf.lat_file, ener_meas_resume
    )
    
    # 3. Lancia il main con lo stato ripristinato
    main_Lang(args)

    return
end
