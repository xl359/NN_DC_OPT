using JLD2, PowerModels, JuMP, Ipopt, LinearAlgebra, Distributions

# auxiliary functions
ns(l) = Int(net[:n_s][l])
nr(l) = Int(net[:n_r][l])
Φ(x) = quantile(Normal(0,1),1-x)
function remove_col_and_row(B,refbus)
    @assert size(B,1) == size(B,2)
    n = size(B,1)
    return B[1:n .!= refbus, 1:n .!= refbus]
end

function build_B̆(B̂inv,refbus)
    Nb = size(B̂inv,1)+1
    B̆ = zeros(Nb,Nb)
    for i in 1:Nb, j in 1:Nb
        if i < refbus && j < refbus
            B̆[i,j] = B̂inv[i,j]
        end
        if i > refbus && j > refbus
            B̆[i,j] = B̂inv[i-1,j-1]
        end
        if i > refbus && j < refbus
            B̆[i,j] = B̂inv[i-1,j]
        end
        if i < refbus && j > refbus
            B̆[i,j] = B̂inv[i,j-1]
        end
    end
    return B̆
end

function load_network_data(caseID)
    data_net = PowerModels.parse_file(caseID)
    # Network size
    G = length(data_net["gen"])
    N = length(data_net["bus"])
    E = length(data_net["branch"])
    D = length(data_net["load"])

    # order bus indexing
    bus_keys=collect(keys(data_net["bus"]))
    bus_keys = bus_keys[sortperm(parse.(Int64, bus_keys))]
    bus_key_dict = Dict()
    for i in 1:N
        push!(bus_key_dict, i => bus_keys[i])
    end
    node(key) = [k for (k,v) in bus_key_dict if v == key][1]

    # Load generation data
    gen_key=collect(keys(data_net["gen"]))
    gen_key = gen_key[sortperm(parse.(Int64, gen_key))]
    p̅ = zeros(G); p̲ = zeros(G); c0 = zeros(G); c1 = zeros(G); c2 = zeros(G); M_p = zeros(N,G)
    for g in gen_key
        p̅[parse(Int64,g)] = data_net["gen"][g]["pmax"]*data_net["baseMVA"]
        p̲[parse(Int64,g)] = data_net["gen"][g]["pmin"]*data_net["baseMVA"]
        if sum(data_net["gen"][g]["cost"]) != 0
            if length(data_net["gen"][g]["cost"]) == 2
                c1[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]
                c0[parse(Int64,g)] = data_net["gen"][g]["cost"][2]
            end
            if length(data_net["gen"][g]["cost"]) == 3
                c2[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]^2
                c1[parse(Int64,g)] = data_net["gen"][g]["cost"][2] / data_net["baseMVA"]
                c0[parse(Int64,g)] = data_net["gen"][g]["cost"][3] 
            end
        end
        M_p[node(string(data_net["gen"][g]["gen_bus"])),parse(Int64,g)] = 1
    end

    # Load demand data
    load_key=collect(keys(data_net["load"]))
    load_key = load_key[sortperm(parse.(Int64, load_key))]

    d = zeros(D); M_d = zeros(N,D)
    for h in load_key
        d[parse(Int64,h)] = data_net["load"][h]["pd"]*data_net["baseMVA"]
        M_d[node(string(data_net["load"][h]["load_bus"])),parse(Int64,h)] = 1
    end

    # Load transmission data
    line_key=collect(keys(data_net["branch"]))
    line_key = line_key[sortperm(parse.(Int64, line_key))]

    β = zeros(E); f̅ = zeros(E); n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    for l in line_key
        β[data_net["branch"][l]["index"]] = -imag(1/(data_net["branch"][l]["br_r"] + data_net["branch"][l]["br_x"]im))
        n_s[data_net["branch"][l]["index"]] = node(string(data_net["branch"][l]["f_bus"]))
        n_r[data_net["branch"][l]["index"]] = node(string(data_net["branch"][l]["t_bus"]))
        f̅[data_net["branch"][l]["index"]] = data_net["branch"][l]["rate_a"]*data_net["baseMVA"]
    end

    # Find reference node
    ref = 1
    for n in 1:N
        if sum(M_p[n,:]) == 0 &&  sum(M_d[n,:]) == 0 == 0
            ref = n
        end
    end

    # Compute PTDF matrix
    B_line = zeros(E,N); B̃_bus = zeros(N,N); B = zeros(N,N)
    for n in 1:N
        for l in 1:E
            if n_s[l] == n
                B[n,n] += β[l]
                B_line[l,n] = β[l]
            end
            if n_r[l] == n
                B[n,n] += β[l]
                B_line[l,n] = -β[l]
            end
        end
    end
    for l in 1:E
        B[Int(n_s[l]),Int(n_r[l])] = - β[l]
        B[Int(n_r[l]),Int(n_s[l])] = - β[l]
    end
    B̃_bus = remove_col_and_row(B,ref)
    B̃_bus = inv(B̃_bus)
    B̃_bus = build_B̆(B̃_bus,ref)
    PTDF = B_line*B̃_bus

    # safe network data
    net = Dict(
    # transmission data
    :f̅ => f̅, :n_s => n_s, :n_r => n_r, :T => PTDF,
    # load data
    :d => d, :M_d => M_d,
    # generation data
    :p̅ => p̅, :p̲ => p̲, :M_p => M_p,
    :c1 => c1, :c2 => c2, :c0=> c0,
    # graph data
    :N => N, :E => E, :G => G, :D => D, :ref => ref,
    )
    return net
end

function solve_OPF(net)
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    JuMP.set_silent(model)
    # model variables
    @variable(model, p[1:net[:G]])
    # model objective
    @objective(model, Min, p'*net[:c2]*p + net[:c1]'p)
    # OPF equations
    @constraint(model, μ, ones(net[:N])'*(net[:M_p]*p .- net[:M_d]*net[:d]) .== 0)
    @constraint(model, μ̅, net[:f̅] .>=  net[:T]*(net[:M_p]*p .- net[:M_d]*net[:d]))
    @constraint(model, μ̲, net[:f̅] .>= -net[:T]*(net[:M_p]*p .- net[:M_d]*net[:d]))
    @constraint(model, net[:p̲] .<= p .<= net[:p̅])
    # solve model
    optimize!(model)
    if "$(termination_status(model))" != "LOCALLY_SOLVED"
        @warn("DC-OPF terminates with status $(termination_status(model))")
    end
    π = JuMP.dual.(μ) .* ones(net[:N]) .- net[:T]'*(JuMP.dual.(μ̅) .- JuMP.dual.(μ̲))
    
    sol = Dict(:status => termination_status(model),
                :obj => JuMP.value.(p)'*net[:c2]*JuMP.value.(p) + net[:c1]'JuMP.value.(p) + sum(net[:c0]),
                :p => JuMP.value.(p),
                :CPUtime => solve_time(model),
                :λ =>  net[:M_d]'*π.*net[:d],
                :π => π)
    return sol
end
