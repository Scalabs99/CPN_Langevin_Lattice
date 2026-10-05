struct CompiledKernel{T <: Integer}
    args::Tuple
    kernel::CUDA.HostKernel
    threads::T
    blocks::T
end

function compile_kernel(fun::Function, fun_args::Tuple, len_arr::T) where {T <: Integer}
    kernel = @cuda launch=false fun(fun_args...)
    config = launch_configuration(kernel.fun)
    threads::T = min(len_arr, config.threads)
    blocks::T = cld(len_arr, threads)
    return CompiledKernel{T}(fun_args, kernel, threads, blocks)
end

@inline function run_kernel(kernel::CompiledKernel)
    kernel.kernel(kernel.args...; threads=kernel.threads, blocks=kernel.blocks)
end
