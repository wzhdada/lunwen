using StableRNGs
using Distributions
using DataStructures
using Printf
using Dates

struct Breakdown <: Event    # machine breaks down 
    id::Int64        
    time::Float64    
end

struct Repair <: Event    # machine is repaired 
    id::Int64        
    time::Float64    
end

struct Arrival <: Event    # lawnmower arrives
    id::Int64         
    time::Float64     
end

struct Departure <: Event    # lawnmower departs
    id::Int64        
    time::Float64    
end

abstract type Event end

mutable struct Entity
    id::Int64                      #  id for each lawnmower
    arrival_time::Float64          # time when the lawnmower arrives to be processed
    start_service_time::Float64    # time when the lawnmower starts service
    completion_time::Float64       # time when the lawnmower is completed
    interrupted::Int64             # times that the machine breaks down when the lawnmower is in service
end
# generate a newly arrived lawnmower where its start_service_time and completion_time are unknown
Entity( id::Int64, arrival_time::Float64 ) = Entity(id, arrival_time, Inf, Inf, 0)

# state
mutable struct State
    time::Float64                               # the system time (simulation time)
    n_entities::Int64                           # the number of entities to have been served
    n_events::Int64                             # tracks the number of events to have occur + queued
    event_queue::PriorityQueue{Event,Float64}   # to keep track of future arrival/departure/breakdown/repair
    waiting_queue::Queue{Entity}                # to keep track of waiting lawnmowers
    in_service::PriorityQueue{Entity,Float64}   # to keep track of lawnmower in service
    machine_status::Int64                       # to keep track of whether the machine is available 
end
# update functions
function update!( system::State, R::RandomNGs, E::Arrival )
    system.n_entities += 1    # new entity will enter the system

    # create an arriving lawnmower and add it to the queue
    lawnmower = Entity(system.n_entities, E.time)
    new_lawnmower = deepcopy(lawnmower)
    enqueue!(system.waiting_queue, new_lawnmower)

    # generate next arrival
    future_arrival = Arrival(system.n_events, system.time + R.interarrival_time())
    enqueue!(system.event_queue, future_arrival, future_arrival.time)

    # if the machine is available, the lawnmower goes to service
    if isempty(system.in_service) && system.machine_status == 0    # machine available and free
        move_to_server!(system,R)
    end
    return new_lawnmower
end

function update!( system::State, R::RandomNGs, E::Departure )
    lawnmower = dequeue!(system.in_service)    # remove lawnmower
    departing_lawnmower = deepcopy(lawnmower)

    if !isempty(system.waiting_queue)    # if someone is waiting, move it to service
        move_to_server!(system, R)
    end
    return departing_lawnmower
end

function update!( system::State, R::RandomNGs, E::Breakdown )
    system.machine_status = 1    # change the status of the machine as breakdown

    # create a repair event for the lawnmower
    future_repair = Repair(system.n_events,system.time + R.repair_time())
    enqueue!(system.event_queue, future_repair, future_repair.time)

    # when the machine breaks down, the time of completion of the current lawnmower will be extended
    if !isempty(system.in_service)
        (breakdown_lawnmower, time) = dequeue_pair!(system.in_service)
        breakdown_lawnmower.completion_time = future_repair.time + time - system.time
        breakdown_lawnmower.interrupted += 1
        enqueue!(system.in_service, breakdown_lawnmower, breakdown_lawnmower.completion_time)
    end
    return nothing
end

function update!( system::State, R::RandomNGs, E::Repair )
    system.machine_status = 0    # change the status of the machine as repaired
    # create a breakdown event for the lawnmower
    future_breakdown = Breakdown(system.n_events, system.time + R.interbreakdown_time())
    enqueue!(system.event_queue, future_breakdown, future_breakdown.time)
    return nothing
end
# parameter structure
struct Parameters
    seed::Int
    mean_interarrival::Float64
    mean_construction_time::Float64
    mean_interbreakdown_time::Float64
    mean_repair_time::Float64
end

# setup random number generators
struct RandomNGs
    rng::StableRNGs.LehmerRNG
    interarrival_time::Function
    construction_time::Function
    interbreakdown_time::Function
    repair_time::Function
end
# construct a function to create all the pieces required
function RandomNGs( P::Parameters )
    rng = StableRNG(P.seed)
    interarrival_time() = rand(rng, Exponential(P.mean_interarrival))
    construction_time() = P.mean_construction_time
    interbreakdown_time() = rand(rng, Exponential(P.mean_interbreakdown_time))
    repair_time() = rand(rng, Exponential(P.mean_repair_time))
    return RandomNGs(rng, interarrival_time, construction_time, interbreakdown_time, repair_time)
end
# initialisation function for the simulation
function initialise( P::Parameters )
    R = RandomNGs( P ) # create the RNGs
    system = State() # create the initial state structure
    # add an arrival at time 0.0
    t0 = 0.0
    system.n_events += 1 # your system state should keep track of events
    enqueue!( system.event_queue, Arrival(0,t0), t0)
    # add a breakdown at time 150.0
    t1 = 150.0
    system.n_events += 1
    enqueue!( system.event_queue, Breakdown(system.n_events, t1), t1 )
    return (system, R)
end

# create an initial (empty) state
function State()
    init_time = 0.0
    init_n_entities = 0
    init_n_events = 0
    init_event_queue = PriorityQueue{Event,Float64}()
    init_waiting_queue = Queue{Entity}()
    init_in_service = PriorityQueue{Entity,Float64}()
    init_machine_status = 0
    return State(init_time, 
           init_n_entities, 
           init_n_events, 
           init_event_queue, 
           init_waiting_queue, 
           init_in_service, 
           init_machine_status)
end

function move_to_server!( system::State, R::RandomNGs )
    # update the lawnmower from the waiting queue
    next_lawnmower = dequeue!(system.waiting_queue)    # remove it from queue
    next_lawnmower.start_service_time = system.time    
    next_lawnmower.completion_time = next_lawnmower.start_service_time + R.construction_time()
    enqueue!(system.in_service, next_lawnmower, next_lawnmower.completion_time) 

    # create departure event 
    system.n_events += 1
    departure_event = Departure(system.n_events, next_lawnmower.completion_time)
    enqueue!(system.event_queue, departure_event, next_lawnmower.completion_time)
end

function run!( system::State, R::RandomNGs, T::Float64, fid_state::IO, fid_entities::IO )
    # the simulation loop
    while system.time<T
        # grab the next event 
        (E, time) = dequeue_pair!(system.event_queue)

        if typeof(E) == Departure && system.machine_status == 1
            (breakdown_lawnmower, delayed_time) = dequeue_pair!(system.in_service)
            enqueue!(system.in_service, breakdown_lawnmower, delayed_time)
            enqueue!(system.event_queue, E, delayed_time)
            continue
        end
        write_state(fid_state, system, E)

        entity = update!(system, R, E)

        if typeof(E) !== Departure
            system.n_events += 1    
        end

        if typeof(E) == Departure 
            write_entity(fid_entities, entity)
        end

        system.time = time    # set the time to the new event
        
    end
    return system
end

# function to writeout parameters
function write_parameters( output::IO, P::Parameters, t::Float64)    
    T = typeof(P)
    for name in fieldnames(T)
        println( output, "# parameter: $name = $(getfield(P,name))" )
    end
    println( output, "# units = minutes")
    println( output, "# T = $t")
end

# function to writeout extra metadata
function write_metadata( output::IO )    
    (path, prog) = splitdir( @__FILE__ )
    println( output, "# file created by code in $(prog)" )
    t = now()
    println( output, "# file created on $(Dates.format(t, "yyyy-mm-dd at HH:MM:SS"))" )
end

# output functions 
function write_state( eve_file::IO, system::State, E::Event; debug_level::Int=0)
    # write the state
    @printf(eve_file, 
            "%12.3f,%6d,%9s,%4d,%4d,%4d,%4d", 
            system.time,
            E.id,
            typeof(E),
            length(system.event_queue),
            length(system.waiting_queue),
            length(system.in_service),
            system.machine_status)
    @printf(eve_file,"\n")
end
function write_entity( entity_file::IO, entity::Entity; debug_level::Int=0)
    # write the entity
    @printf(entity_file, 
            "%4d,%12.3f,%12.3f,%12.3f,%4d", 
            entity.id,
            entity.arrival_time,
            entity.start_service_time,
            entity.completion_time,
            entity.interrupted)
    @printf(entity_file,"\n")
end