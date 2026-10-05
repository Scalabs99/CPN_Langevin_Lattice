using Dates

include("CPN_LangRK.jl")

function print_usage()
    println("Usage: $PROGRAM_FILE start config.toml")
    spaces = ' ' ^ length("Usage: ")
    println(spaces, "to start from scratch")
    println(spaces, "Required keys in config file:")
    println(spaces, required_config_keys)
    println(spaces, "See simulation_configurations/sample_configuration.toml for further details.")
    println("   or: $PROGRAM_FILE load checkpoint.jld")
    println(spaces, "to resume from a previous checkpoint")
end

function parse_args(args)
    if length(args) == 2
        if lowercase(args[1]) in ["start", "s"]
            launch_main_Lang(args[2])
            return
        elseif lowercase(args[1]) in ["load", "l", "resume", "r"]
            resume_main_Lang(args[2])
            return
        end
    end
    print_usage()
    return
end

# ---- Program entry point ----

println(now(), ": Library compilation successful. Starting execution.")
parse_args(ARGS)
println(now(), ": Execution terminated.")
