using JLD2,JuMP, Gurobi,LinearAlgebra, Random
using Pkg
Pkg.activate() 
const GRB_ENV = Gurobi.Env()

# this fector is how much you want to expand the ρ value. 
# it is defaulted to 1
expand_factor = 1

# this factor is used to fine tune the ρ⋆ value so that the complementarity condition is satisfied
ρ_factor = 1.5
include("find_rho.jl")
ρ  = rho_value()

ρ = ρ*ρ_factor*expand_factor
include("aux_fun.jl")

# load OPF data (traning set and network params)
data = JLD2.load("train_data_paper.jld2", "train_data")
net = data[:net]
# load neural network
nn = JLD2.load("nn_trained_paper.jld2", "nn_trained")

# nn size
n_in, n_out = 3, 3
n_hl, n_hn = length(nn[:W])-1, size(nn[:W][2],1)

dc_load_scale = 0.9


function nn_sos1_ref(d)
    """
        This function takes in a value of datacenter load and compute the solution of the original problem 
        without the datacenter load constraints. It is equavaliant to passing it through the neural network.

        Arg: 

            d: data center load

        Return:

            :λ̂  : Charge of the datacenter loads
            :y  : The hidden neuron values
            :v  : Result of the ReLU function properity
        """
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(model, y[1:n_hn,1:n_hl]>=0)
    @variable(model, v[1:n_hn,1:n_hl]>=0)
    @variable(model, λ̂[1:n_out])

    # mixed-integer reformulation of the trained neural network
    @constraint(model, y[:,1] .== nn[:W][1]*d      .+ nn[:b][1] .+ v[:,1])
    @constraint(model, y[:,2] .== nn[:W][2]*y[:,1] .+ nn[:b][2] .+ v[:,2])
    @constraint(model,      λ̂ .== nn[:W][3]*y[:,2] .+ nn[:b][3])
    for i in 1:n_hn, k in 1:n_hl
        @constraint(model, [v[i,k],y[i,k]] in SOS1())
    end

    optimize!(model)
    return Dict(:λ̂ => JuMP.value.(λ̂), :y => JuMP.value.(y), :v => JuMP.value.(v))
end

# function to normalize and denormalize loads and charges
# un_normalize load
_d_(d) = d .*(data[:d_max]-data[:d_min]) .+ data[:d_min]
# unnormalize charge
_λ_(λ) = λ .*(data[:λ_max]-data[:λ_min]) .+ data[:λ_min]
# normalize load
_d̂_(d) = (d .- data[:d_min])./(data[:d_max]-data[:d_min])
# normalize charge
_λ̂_(λ) = (λ .- data[:λ_min])./(data[:λ_max]-data[:λ_min])

Δ =  sum(dc_load_scale*_d̂_(net[:d]))
# compute the upper limit
_d̂_lower = 0.8*_d̂_(net[:d])
_d̂_upper = _d̂_(net[:d])


function opt_sos1()
    """

        This function computes the solution of the original problem 
        without the datacenter load constraints. It is equavaliant to passing it through the neural network.


        Return:
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

    @objective(model, Min, ones(n_out)'*λ̂)

    @constraint(model, d̂ .<= _d̂_(net[:d]))
    @constraint(model, d̂ .>= 0.8*_d̂_(net[:d]))
    @constraint(model, ones(n_in)'*d̂ == sum(dc_load_scale*_d̂_(net[:d])))

    # mixed-integer reformulation of the trained neural network
    @constraint(model, y[:,1] .== nn[:W][1]*d̂      .+ nn[:b][1] .+ v[:,1])
    @constraint(model, y[:,2] .== nn[:W][2]*y[:,1] .+ nn[:b][2] .+ v[:,2])
    @constraint(model,      λ̂ .== nn[:W][3]*y[:,2] .+ nn[:b][3])
    for i in 1:n_hn, k in 1:n_hl
        @constraint(model, [v[i,k],y[i,k]] in SOS1())
    end

    optimize!(model)
    return Dict(:y => JuMP.value.(y), :v => JuMP.value.(v), :d̂_act => _d_(JuMP.value.(d̂)), :λ̂_act => _λ_(JuMP.value.(λ̂)), :d̂ => JuMP.value.(d̂), :λ̂ => JuMP.value.(λ̂), :obj => JuMP.objective_value(model))
end

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

    @constraint(model, d̂ .<= _d̂_(net[:d]))
    @constraint(model, d̂ .>= 0.8*_d̂_(net[:d]))
    @constraint(model, ones(n_in)'*d̂ == sum(dc_load_scale*_d̂_(net[:d])))

    # mixed-integer reformulation of the trained neural network
    @constraint(model, y[:,1] .== nn[:W][1]*d̂      .+ nn[:b][1] .+ v[:,1])
    @constraint(model, y[:,2] .== nn[:W][2]*y[:,1] .+ nn[:b][2] .+ v[:,2])
    @constraint(model,      λ̂ .== nn[:W][3]*y[:,2] .+ nn[:b][3])

    optimize!(model)
    return Dict(:y => JuMP.value.(y), :v => JuMP.value.(v), :d̂_act => _d_(JuMP.value.(d̂)), :λ̂_act => _λ_(JuMP.value.(λ̂)), :d̂ => JuMP.value.(d̂), :λ̂ => JuMP.value.(λ̂), :obj => JuMP.objective_value(model))
end

sol_sos1 = opt_sos1()

# Helper functions 
# penality function
ϕ(y,v) = 1/4*norm(y+v,2)^2 - 1/4*norm(y-v,2)^2
# total charge computation function
f(λ̂,y,v) = ones(n_out)'_λ_(λ̂) + ρ*sum(ϕ(y[:,i],v[:,i]) for i in 1:n_hl)
# complementary condition violatioin computation
com_cond(y,v) = sum(y[:,i]'v[:,i] for i in 1:n_hl)


# compute initial ỹ and ṽ
# take the third sample to be the initial d value 
sample = 3
sol_init = nn_sos1_ref(data[:d][:,sample])

λ̂ =sol_init[:λ̂]
ỹ = sol_init[:y]
ṽ = sol_init[:v]
d̃ = zeros(net[:D])
f_aft = 0

# start the DCA algorithm iteration
ϵ_tol = 1e-5
dca_charges = []
sos1 = []
iteration_number = 0
for i in 1:100000
    global λ̂,ỹ,ṽ,iteration_number,d̃,f_aft
    iteration_number = iteration_number + 1
    f_pre = f(λ̂,ỹ,ṽ)
    sol_dca_sub = opt_dca_sub(ỹ,ṽ)
    ỹ, ṽ, λ̂, d̃ = sol_dca_sub[:y], sol_dca_sub[:v], sol_dca_sub[:λ̂], sol_dca_sub[:d̂_act]
    f_aft = f(λ̂,ỹ,ṽ)
    i % 500 == 0 ? println("i: $i ... f: $(f(λ̂,ỹ,ṽ)) ... comp_viol: $(com_cond(ỹ,ṽ)) ... Δf: $(norm(f_aft-f_pre))") : NaN
    norm(f_aft-f_pre) <= ϵ_tol ? break : NaN 
    append!(dca_charges,f_aft)
    append!(sos1,com_cond(ỹ,ṽ))
end


data_set = Dict(:ρ => ρ,
                  :dca_charges => dca_charges,
                  :ground_truth_charge_final =>sum(sol_sos1[:λ̂_act]),
                  :dca_charges_final => f(λ̂,ỹ,ṽ),
                  :sos1_cond => sos1,
                  :iteration_number => iteration_number
                    )
@save "dca_charges.jld2" data_set

charge_1 = data_set[:dca_charges]./data_set[:ground_truth_charge_final]
p1 = plot(charge_1,  label= "ρ⋆",color= "black" ,linestyle=:solid, lw = 3)

plot!(ones(10000), label="ground truth",linestyle=:dash, color = :black)

plot!(size=(600,250))
plot!(xlabel="iteration", ylabel="normalized total charge",  # Common labels
xaxis = :log,
ylims=(0.999, 1.0015),
xlim=(10, 10000),
legend=:bottomleft)
plot!(tickfontsize=10) 
plot!(legendfontsize=8)
plot!(ylabelfontsize=11)
plot!(xlabelfontsize=11)
plot!(dpi=300)

savefig(p1, "smallcase_convergence.png")




