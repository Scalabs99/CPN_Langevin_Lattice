using CUDA
using Serialization

include("read_write_funcs.jl")

mutable struct OMF_args_copies{F <: AbstractFloat, I <: Integer, I2 <: Integer}
    const Npoint::I
    const n_meas::I
    const NHMC::I
    const dt::F
    const n_comps::I
    const max_ptord::I
    const measure_every::I
    const n_copies::I
    const cuda_rng::CUDA.RNG
    const nhmc_rng::Random.TaskLocalRNG
    iter_start::I2
    x::CuArray{F, 5}
end

function OMF_args(Npoint::I, n_meas::I, NHMC::I, dt::F, n_comps::I, max_ptord::I, measure_every::I, cuda_rng::CUDA.RNG, nhmc_rng::Random.TaskLocalRNG, iter_start::I2, 
    x::CuArray{F, 4}) where {F <: AbstractFloat, I <: Integer, I2 <: Integer}
    return (Npoint=Npoint, n_meas=n_meas, NHMC=NHMC, dt=dt, n_comps=n_comps, max_ptord=max_ptord, measure_every=measure_every, cuda_rng=cuda_rng, nhmc_rng=nhmc_rng,
    iter_start=iter_start, x=x)
end

function get_energy_filename(checkpt_fname::String)
    fname = "../energies" * checkpt_fname[findlast('/', checkpt_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "txt" # changes the extension
    return fname
end

function get_lat_checkpoint(lat_fname::String)
    fname = "../checkpoint" * lat_fname[findlast('/', lat_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function adapt_old_checkpoint_copies(old_checkpoint_fname::String, lat_fname::String=""; lat_chkpt::String="", config_fname::String="", max_saving_time::Int=600)
    new_checkpoint_fname = old_checkpoint_fname[1:end-4] * "_new.jld"
    lat_present = false
    if lat_fname != ""
        lat_chkpt = lat_chkpt != "" ? lat_chkpt : get_lat_checkpoint(lat_fname)
        if isfile(lat_chkpt)
            println("Energy site-by-site checkpoint found.")
            lat_present = true
        else
            error("Energy site-by-site checkpoint NOT found: $lat_chkpt")
        end
    end
    old_args::OMF_args_copies = deserialize(old_checkpoint_fname)
    new_args = OMF_args(
        old_args.Npoint,
        old_args.n_meas,
        old_args.NHMC,
        old_args.dt,
        old_args.n_comps,
        old_args.max_ptord,
        old_args.measure_every,
        old_args.n_copies,
        old_args.cuda_rng,
        old_args.nhmc_rng,
        get_energy_filename(old_checkpoint_fname),
        config_fname,
        new_checkpoint_fname,
        max_saving_time,
        typeof(old_args.Npoint)(old_args.iter_start),
        old_args.x,
        lat_present ? lat_fname : nothing,
        lat_present ? deserialize(lat_chkpt) : nothing
    )
    serialize(new_checkpoint_fname, new_args)
    println("New checkpoint saved to file: ", new_checkpoint_fname)
    return
end

function adapt_old_checkpoint(old_checkpoint_fname::String, lat_fname::String=""; lat_chkpt::String="", config_fname::String="", max_saving_time::Int=600)
    new_checkpoint_fname = old_checkpoint_fname[1:end-4] * "_new.jld"
    lat_present = false
    if lat_fname != ""
        lat_chkpt = lat_chkpt != "" ? lat_chkpt : get_lat_checkpoint(lat_fname)
        if isfile(lat_chkpt)
            println("Energy site-by-site checkpoint found.")
            lat_present = true
        else
            error("Energy site-by-site checkpoint NOT found: $lat_chkpt")
        end
    end
    old_args = deserialize(old_checkpoint_fname)
    new_args = OMF_args(
        old_args.Npoint,
        old_args.n_meas,
        old_args.NHMC,
        old_args.dt,
        old_args.n_comps,
        old_args.max_ptord,
        old_args.measure_every,
        one(old_args.n_comps),
        old_args.cuda_rng,
        old_args.nhmc_rng,
        get_energy_filename(old_checkpoint_fname),
        config_fname,
        new_checkpoint_fname,
        max_saving_time,
        typeof(old_args.Npoint)(old_args.iter_start),
        reshape(old_args.x, size(old_args.x)..., 1),
        lat_present ? lat_fname : nothing,
        lat_present ? reshape(old_ener_meas, (size(old_ener_meas)[1:3]..., 1, size(old_ener_meas)[4:end]...)) : nothing
    )
    serialize(new_checkpoint_fname, new_args)
    println("New checkpoint saved to file: ", new_checkpoint_fname)
    return
end
