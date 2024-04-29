using CSV, DataFrames

# include the simulation code
include("valet.jl")

# get parameters from a file
valet_parameters = DataFrame(CSV.File("valet_parameters.csv",comment="#"))

# choose a final time
final_time = 10_000.0

# choose some seed values
seeds = 1:100

# rerun simulation if output already exists?
rerun = false

# "Simulation harness" ... loop over seeds and parameters values
for seed in seeds
    for k in 1:nrow(valet_parameters)
        n_valets = valet_parameters.n_valets[k]
        mean_interarrival = valet_parameters.mean_interarrival[k]
        service_time = valet_parameters.service_time[k]
        #final_time = valet_parametres.final_time[k]
        P = Parameters(seed,n_valets,mean_interarrival,service_time,final_time)
        if rerun || any(.!isfile.(output_files(P)))
            run_valet_sim(P)
        end
    end
end