
using JLD2
using JuMP, Gurobi,Random
using LinearAlgebra
using Plots,StatsBase
using JLD2
using LinearAlgebra
using Statistics
using Plots

using Colors
const GRB_ENV = Gurobi.Env()

purple=RGB(122/256,30/256,71/256)
blue=RGB(0/256, 39/256, 76/256)
green=RGB(74/256, 103/256, 65/256)

include("aux_fun_large.jl")
include("find_rho_large.jl")

ρ  = rho_value()

sample_index = 28
Random.seed!(100)
n_d = 10000 # specify the number of d we want to obtain
# load the testing data as baselines
dataset = JLD2.load("dataset.jld2", "dataset")
d_test = dataset[:test][:d_test]
n_test = size(d_test,2)
# select a set of random numbers 
sampled_numbers = sample(1:n_test, 100, replace=false)
# choose a number from the random number set as the baseline sample selection
# this is the index where Δ is from
sample_ind = sampled_numbers[sample_index]
net = dataset[:net]
# load neural network
nn = JLD2.load("nn_trained.jld2", "nn_trained")

# nn size
n_in, n_out = size(nn[:W][1],2), size(nn[:W][length(nn[:W])],1)
n_hl, n_hn = length(nn[:W])-1, size(nn[:W][2],1)


function solve_init_6(DNN,x)
    """
    Calculate the initial points for approximation method.

    # Arguments
    - DNN: the deep neural network we try to validate
    - x: a sample input from the permissible input space

    # Returns
    - the initial points for x and y
    """

    y1_prim = DNN[:W][1]*x + DNN[:b][1]
    y1 = max.(y1_prim,0)
    y2_prim = DNN[:W][2]*y1 + DNN[:b][2]
    y2 = max.(y2_prim,0)
    y3_prim = DNN[:W][3]*y2 + DNN[:b][3]
    y3 = max.(y3_prim,0)
    y4_prim = DNN[:W][4]*y3 + DNN[:b][4]
    y4 = max.(y4_prim,0)
    y5_prim = DNN[:W][5]*y4 + DNN[:b][5]
    y5 = max.(y5_prim,0)
    λ̂  = DNN[:W][6]*y5 + DNN[:b][6]
    y_prim = hcat(y1_prim, y2_prim,y3_prim,y4_prim,y5_prim)
    y = abs.(max.(y_prim,zeros(size(y_prim))))
    v = abs.(-min.(y_prim,zeros(size(y_prim))))
 
    return  Dict(:λ̂ => λ̂ , :y => y, :v => v)
end

# function to normalize and denormalize loads and charges
_d_(d) = d .*(dataset[:norm_param][:d_max]-dataset[:norm_param][:d_min]) .+ dataset[:norm_param][:d_min]
_λ_(λ) = λ .*(dataset[:norm_param][:λ_max]-dataset[:norm_param][:λ_min]) .+ dataset[:norm_param][:λ_min]
_d̂_(d) = (d .- dataset[:norm_param][:d_min])./(dataset[:norm_param][:d_max]-dataset[:norm_param][:d_min])
_λ̂_(λ) = (λ .- dataset[:norm_param][:λ_min])./(dataset[:norm_param][:λ_max]-dataset[:norm_param][:λ_min])

function opt_dca_sub(ỹ,ṽ)
    """
    This function takes in the initial value of the hidden neurons and the v vector, which 
    and returns the dca solution of each iteration_number

    Arg: 
        ỹ:initial neuron value
        ṽ: initial v value
    
    Return
        :y  : The hidden neuron values
        :v  : Result of the ReLU function properity
        :d̂_act: computed optimal load
        :λ̂_act: computed optimal charge
        :d̂ : normalized optimal load
        :λ̂ : normalized optimal charge
        :obj: sum of the output of the nenral network
    """
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    JuMP.set_silent(model)

    @variable(model, y[1:n_hn,1:n_hl]>=0)
    @variable(model, v[1:n_hn,1:n_hl]>=0)
    @variable(model, d̂[1:n_in])
    @variable(model, λ̂[1:n_out])

    @objective(model, Min, ones(n_out)'*λ̂
                                + ρ/4*sum((y[:,i] .+ v[:,i])'*(y[:,i] .+ v[:,i]) for i in 1:n_hl)
                                - ρ/2*sum((ỹ[:,i] .- ṽ[:,i])'*(y[:,i] .- v[:,i]) for i in 1:n_hl))

    @constraint(model, d̂ .<= _d̂_upper)
    @constraint(model, d̂ .>= _d̂_lower)
    @constraint(model, ones(n_in)'*d̂ == Δ)

    # mixed-integer reformulation of the trained neural network
    @constraint(model, y[:,1] .== nn[:W][1]*d̂      .+ nn[:b][1] .+ v[:,1])
    @constraint(model, y[:,2] .== nn[:W][2]*y[:,1] .+ nn[:b][2] .+ v[:,2])
    @constraint(model, y[:,3] .== nn[:W][3]*y[:,2] .+ nn[:b][3] .+ v[:,3])
    @constraint(model, y[:,4] .== nn[:W][4]*y[:,3] .+ nn[:b][4] .+ v[:,4])
    @constraint(model, y[:,5] .== nn[:W][5]*y[:,4] .+ nn[:b][5] .+ v[:,5])
    @constraint(model,      λ̂ .== nn[:W][6]*y[:,5] .+ nn[:b][6])

    optimize!(model)
    return Dict(:y => JuMP.value.(y), :v => JuMP.value.(v), :d̂_act => _d_(JuMP.value.(d̂)), :λ̂_act => _λ_(JuMP.value.(λ̂)), :d̂ => JuMP.value.(d̂), :λ̂ => JuMP.value.(λ̂), :obj => JuMP.objective_value(model))
end

# Helper functions 
# penality function
ϕ(y,v) = 1/4*norm(y+v,2)^2 - 1/4*norm(y-v,2)^2
# total charge computation function
f(λ̂,y,v) = ones(n_out)'_λ_(λ̂) + ρ*sum(ϕ(y[:,i],v[:,i]) for i in 1:n_hl)
# complementary condition violatioin computation
com_cond(y,v) = sum(y[:,i]'v[:,i] for i in 1:n_hl)

# take the baseline distribution
x̃ = d_test[:,sample_ind]
Δ =  sum(x̃)# change it to each sample of net[:dc] from input of neuron network

# compute the upper limit

_d̂_lower = 0.1*_d̂_(net[:dc])
_d̂_upper = 1.9*_d̂_(net[:dc])

# define baseline
baseline_load = x̃

# retrieve the initial solution for DCA iteration
sol_init = solve_init_6(nn,x̃);
ỹ = sol_init[:y]
ṽ = sol_init[:v]

f_aft = 0 # set the current solution total charge to be 0
pre = 0 # set the previous solution total charge to be 0
iteration_number = 0 # track the iteration number
ϵ_tol = 1e-5 # set the terminating condition
tot_iter = 500000 # set the total number of iteration

dca_charge = []
comp_violation = []

# DCA iteration
elapsed_time = @elapsed for i in 1:tot_iter
    global λ̂,ỹ,ṽ,d̃,ρ,pre,nn_sol,iteration_number,f_aft
    # compute the dca objective at each iteration
    sol_dca_sub = opt_dca_sub(ỹ,ṽ)
    ỹ, ṽ, λ̂, d̃ = sol_dca_sub[:y], sol_dca_sub[:v], sol_dca_sub[:λ̂], sol_dca_sub[:d̂_act] # note here the d is the actula d
    # compute the total charge
    f_aft = f(λ̂,ỹ,ṽ)

    append!(dca_charge,f_aft)
    append!(comp_violation,com_cond(ỹ,ṽ) )
    i % 500 == 0 ? println("i: $i...f: $(round((f(λ̂,ỹ,ṽ))))...comp: $(com_cond(ỹ,ṽ)) ... Δf: $(norm(f_aft-pre))") : NaN
    if  norm(f_aft-pre) <= ϵ_tol 
        iteration_number = i
        break
    end
    if i == tot_iter
        iteration_number = i 
    end
    pre   = f_aft
end

dca_load = d̃
@show elapsed_time
@show iteration_number
@show ρ
base_net = deepcopy(net);
base_net[:dc] = _d_(baseline_load);
sol_opf_base = solve_OPF(base_net);
baseline_total_charge = sum(sol_opf_base[:λ])
@show baseline_total_charge

# dca total charge
new_net = deepcopy(net);
new_net[:dc] = d̃;
sol_opf_new = solve_OPF(new_net);
dca_total_charge = sum(sol_opf_new[:λ])
@show dca_total_charge

# Locational Marginal Price
LMP = sol_opf_new[:π]
@show LMP


dca_data = Dict(:elapsed_time => elapsed_time,
                      :iteration_number => iteration_number,
                      :ρ => ρ,
                      :baseline_load => baseline_load,
                      :dca_load => dca_load,
                      :baseline_total_charge => baseline_total_charge,
                      :dca_total_charge => dca_total_charge,
                      :dca_charge => dca_charge,
                      :comp_violation => comp_violation
)



@save "dca_data.jld2" dca_data

p = plot(dca_charge./dca_total_charge, frame=:box,xlabel="iteration", ylabel="normalized total charge",
xaxis = :log,
       label = false,
        lw=0.8, alpha=0.3,
        xticks=10 .^ (0:5),
       color=green)
       plot!(size=(600,250))

plot!(dpi=300)

plot!(xlim=(1, 1e5+30000))
plot!(ylim=(1, 2.4))

plot!(tickfontsize=10) 
plot!(legendfontsize=10)
plot!(ylabelfontsize=11)
plot!(xlabelfontsize=11)
savefig(p, "convergence.png")

