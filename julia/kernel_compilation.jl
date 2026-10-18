struct CompiledKernel{ArgsT <: Tuple, DimsT <: Tuple}
    args::ArgsT
    kernel::CUDA.HostKernel
    threads::DimsT
    blocks::DimsT
end

function compile_kernel(fun::Function, fun_args::Tuple, len_arrx::T, len_arry::T) where {T <: Integer}
    kernel = @cuda launch=false fun(fun_args...)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads 

    # 2. Distribuzione dinamica:
    # Diamo prima più thread possibili all'asse X (i siti spaziali, ottimizza la memoria)
    tx::T = min(len_arrx, max_threads)
    
    # Quello che avanza del limite lo diamo all'asse Y (le copie)
    ty::T = min(len_arry, max(one(T), max_threads ÷ tx))

    # 3. Calcolo dinamico dei blocchi per coprire l'intera griglia
    bx::T = cld(len_arrx, tx)
    by::T = cld(len_arry, ty)
    
    threads_2d = (tx, ty)
    blocks_2d = (bx, by)

    return CompiledKernel{typeof(fun_args), typeof(threads_2d)}(fun_args, kernel, threads_2d, blocks_2d)
end

@inline function run_kernel(kernel::CompiledKernel)
    kernel.kernel(kernel.args...; threads=kernel.threads, blocks=kernel.blocks)
end
