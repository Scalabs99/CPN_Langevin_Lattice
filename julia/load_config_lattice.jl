using TOML

# Required fields
required_config_keys = [
    "n_side_sites", "n_measurements", "dt", "n_indep_comps",
    "max_perturbative_order", "energy_filename"
] # don't add vacuum filename to the required keys. 
  # This way, when the name of the instanton file is not provided, the program runs using the trivial vacuum (nu = 0 and zeta = 1)
function is_file_writeable(filepath::String)
    # Helper function to extend compatibility for Julia v. < 1.11
    function is_writable(path)
        if hasmethod(iswritable, (String,))
            return iswritable(path)
        else
            return (Sys.uperm(path) & 0x02) != 0
        end
    end

    if ispath(filepath)
        # File exists - check if it's writable
        return is_writable(filepath)
    else
        # File doesn't exist - check if parent directory is writable
        parent_dir = dirname(filepath)
        if isempty(parent_dir)
            parent_dir = "."
        end
        return ispath(parent_dir) && is_writable(parent_dir)
    end
end

function check_required_keys(config_fname, check_chkpt_fname=true)
    cfg = TOML.parsefile(config_fname)
    for key in required_config_keys
        haskey(cfg, key) || error("Missing required config key: $key")
    end
    if check_chkpt_fname && !haskey(cfg, "checkpoint_filename")
        error("Missing required config key: checkpoint_filename")
    end
    return
end

function parse_config_file(fname, checkpt_fname=nothing)
    cfg = TOML.parsefile(fname)

    # get number of bits for ints and floats from config file, if they exist
    intBits   = get(cfg,   "intBits", "32")
    floatBits = get(cfg, "floatBits", "32")
    intType   =   intBits == "64" ?   Int64 :   Int32
    floatType = floatBits == "64" ? Float64 : Float32
    if (intType == Int32 && floatType == Float64) || (intType == Int64 && floatType == Float32)
        println("[WARNING]: GPU int variables will be $intBits bits, while float variables will be $floatBits bits.")
        println("           This can cause performance issues.")
    end

    # Mandatory fields
    Npoint            = intType(cfg["n_side_sites"])
    n_meas            = intType(cfg["n_measurements"])
    dt                = floatType(cfg["dt"])
    n_comps           = intType(cfg["n_indep_comps"])
    max_ptord         = intType(cfg["max_perturbative_order"])
    en_fname          = cfg["energy_filename"]
    is_file_writeable(en_fname) || error("Energy file $en_fname is not writeable")
    if checkpt_fname == nothing
        checkpt_fname = cfg["checkpoint_filename"]
    end
    is_file_writeable(checkpt_fname) || error("Checkpoint file $checkpt_fname is not writeable")

    # Optional fields
    measure_every     = intType(get(cfg, "measure_every", 1))
    n_copies          = intType(get(cfg, "n_copies", 1))
    if haskey(cfg, "energy_site_by_site_matlab")
        lat_file      = cfg["energy_site_by_site_matlab"]
        is_file_writeable(lat_file) || error("Energy site-by-site file ", lat_file, " is not writeable")
    else
        lat_file = nothing
    end
    cuda_seed         = Int(get(cfg, "cuda_seed", 0))
    max_saving_time   = Int(get(cfg, "max_saving_time", 600))

    vac_fname         = get(cfg, "vacuum_filename", "")

    return (
        Npoint = Npoint, n_meas = n_meas, dt = dt, n_comps = n_comps, max_ptord = max_ptord,
        en_fname = en_fname, checkpt_fname = checkpt_fname, measure_every = measure_every, n_copies = n_copies,
        lat_file = lat_file, cuda_seed = cuda_seed, max_saving_time = max_saving_time, vac_fname = vac_fname
    )
end
