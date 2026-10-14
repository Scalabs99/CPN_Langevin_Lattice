struct CompiledKernel{T <: Integer, ArgsT <: Tuple}
    args::Tuple
    kernel::CUDA.HostKernel
    threads::T
    blocks::T
    stream::CUDA.CuStream
end

function compile_kernel(fun::Function, fun_args::Tuple, len_arr::T, stream::CUDA.CuStream=CUDA.default_stream()) where {T <: Integer}
    kernel = @cuda launch=false fun(fun_args...)
    config = launch_configuration(kernel.fun)
    threads::T = min(len_arr, config.threads)
    blocks::T = cld(len_arr, threads)
    return CompiledKernel{T, typeof(fun_args)}(fun_args, kernel, threads, blocks, stream)
end

@inline function run_kernel(kernel::CompiledKernel)
    kernel.kernel(kernel.args...; threads=kernel.threads, blocks=kernel.blocks, stream=kernel.stream)
end
