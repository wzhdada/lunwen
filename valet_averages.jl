using CSV, DataFrames, Statistics

# calculate the time average of the number of customers waiting for one simulation:

valet_data = DataFrame(CSV.File("data/n_valets3/mean_interarrival2/service_time5/seed1/state.csv",comment="#"))
valet_data[:,:n_waiting]=max.(0,valet_data[:,:n_customers].-3)
time_average = mean(valet_data[:,:n_waiting])

# ---

# calculate the ensemble average at t=10000:

# write a little function to get number of cars waiting in queue at t=10000
function get_n_waiting_end(seed)
    valet_data = DataFrame(CSV.File("data/n_valets3/mean_interarrival2/service_time5/seed"*string(seed)*"/state.csv",comment="#"))
    final_queue_size = max(0,valet_data[end-1,:n_customers]-3)
    return final_queue_size
end

valet_ensemble_average = mean(get_n_waiting_end.(1:100))

# ---

# calculate the time average for all simulations

valet_time_averages = DataFrame()
valet_time_averages[:,:seed] = 1:100

# write a little function to do this (hopefully more efficient than a loop)

function get_valet_time_average(seed)
    valet_data = DataFrame(CSV.File("data/n_valets3/mean_interarrival2/service_time5/seed"*string(seed)*"/state.csv",comment="#"))
    valet_data[:,:n_waiting]=max.(0,valet_data[:,:n_customers].-3)
    time_average = mean(valet_data[:,:n_waiting])
    return time_average
end

valet_time_averages[:,:time_average] = get_valet_time_average.(valet_time_averages.seed)

# and take the mean across all seeds:

mean(valet_time_averages.time_average)

