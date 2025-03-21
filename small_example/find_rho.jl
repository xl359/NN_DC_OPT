using JLD2
using JuMP, Gurobi
using LinearAlgebra
using Plots
const GRB_ENV = Gurobi.Env()


function rho_value(dc_load_scale = 0.9, n_in = 3, n_out = 3)
    """

    This function takes in the loading scale and the input output dimension of the neural network and 
        return the optimal rho value for DCA computation

    Arg:

            dc_load_scale: loading scale
            n_in: input dimension
            n_out: output dimension 
    Return: 
            rho value

    """
    # load OPF data (traning set and network params)
    data = JLD2.load("train_data_paper.jld2", "train_data")
    net = data[:net]
    # load neural network
    nn = JLD2.load("nn_trained_paper.jld2", "nn_trained")

    #dc_load_scale = 0.9

    # nn size
    #n_in, n_out = 3, 3
    n_hl, n_hn = length(nn[:W])-1, size(nn[:W][2],1)

    # function to normalize and denormalize loads and charges
    _d_(d) = d .*(data[:d_max]-data[:d_min]) .+ data[:d_min]
    _λ_(λ) = λ .*(data[:λ_max]-data[:λ_min]) .+ data[:λ_min]
    _d̂_(d) = (d .- data[:d_min])./(data[:d_max]-data[:d_min])
    _λ̂_(λ) = (λ .- data[:λ_min])./(data[:λ_max]-data[:λ_min])


    Δ =  sum(dc_load_scale*_d̂_(net[:d]))
    # compute the upper limit
    _d̂_lower = 0.8*_d̂_(net[:d])
    _d̂_upper = _d̂_(net[:d])

    function computing_matricies(n_hl, n_hn,nn, Δ, _d̂_lower, _d̂_upper)
         """
         this function inherent the neural network parameter computed above
            and return several matricies to present the formulation in the
            vectorized formulation
        Arg: 

            n_hl: number of hidden layers
            n_hn: number of nidden neurons of each layer
            nn: neural network
            Δ: summation constraint on d
            _d̂_lower: _d̂_upper: box constraints on the data center loads

        Return: 
            param: a dictionary containing all the necessary matricies
         """


        W = float.(Matrix(I, n_hl*n_hn, n_hl*n_hn))
        for i = 2:n_hl
            W[((i-1)*n_hn+1):i*n_hn, ((i-2)*n_hn+1):(i-1)*n_hn]=-nn[:W][i]
        end
        # compute b
        b = []
        for i = 1:n_hl
            b = [b;-nn[:b][i]]
        end

        # compute f
        f_  = [Δ;-Δ;_d̂_lower;-_d̂_upper]

        if n_hl > 1
            param = Dict("c" => [zeros(1,n_hn*(n_hl-1)) ones(n_out)'*nn[:W][n_hl+1]]', 

                        "A" => [ones(n_in)';-ones(n_in)';Matrix(I, n_in, n_in);-Matrix(I, n_in, n_in)],

                        "f" => f_,

                        "V" => [-nn[:W][1];zeros(n_hn*(n_hl-1),n_in)],
                                
                        "W" => W,

                        "b" => b) 
        end
        return param
    end
    param = computing_matricies(n_hl, n_hn,nn, Δ, _d̂_lower, _d̂_upper)

    function feasible_d(_d̂_lower,_d̂_upper,Δ)
        """

            This function computes a feasible load value such that it satisfies the 
            box constraint on the load and the summation of the load = Δ
        
        Arg: 
                _d̂_lower: lower bound on load
                _d̂_upper: upper bound on load
                Δ       : sum of the load constraint

        Return: 
                :d  : load of datacenter. We use it as the starting point of DCA computation. 
                
        """

        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(model)
        # specifying variable
        @variable(model,d[1:n_in])
        # specifying constrains
        @constraint(model, d .>=_d̂_lower)
        @constraint(model, d .<=  _d̂_upper)
        @constraint(model, dot(ones(1,length(d)),d) == Δ)
        optimize!(model)
        return Dict(:d => value.(d))
    end

    # compute a initialization of load value such that it satisfies the loading constraints
    d_init = feasible_d(_d̂_lower,_d̂_upper,Δ)

    function nn_sos1_ref_rho(d)

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
        JuMP.set_silent(model)
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

    # compute a point with no y = w = 0
    d̃ = d_init[:d]
    sol_init = nn_sos1_ref_rho(d̃);
    ỹ = vec(sol_init[:y])
    ṽ = vec(sol_init[:v])
    sumcond = sum(ỹ .== ṽ)
    if sumcond != 0
        println("warning")
    end

    # find the index set:
    function index_set(ỹ, ṽ)

         """
        
        This function computes the index sets corresponding to the RxLP solution
        
        Arg:
            ỹ: y coordinate of the feasble point computed using feasible_d
            ṽ: v coordinate of the feasble point computed using feasible_d
        Return: 
            Iy, Iv, I0 index sets for solving RxLP
         """
            
        Iy = []
        Iv = []
        I0 = []

        for i = 1:length(ỹ)
        # global Iy,Iv,I0

            if ỹ[i] == 0 && ṽ[i] > 0
                Iy = [Iy;i]
            end
            if ỹ[i] > 0 && ṽ[i] == 0
                Iv = [Iv;i]
            end
            if ỹ[i] == 0 && ṽ[i] == 0
                I0 = [I0;i]
            end
        end
        return Iy, Iv, I0 
    end

    Iy, Iv, I0 = index_set(ỹ, ṽ)

    function solve_RxLP(param,Iy,Iv) 

       """

            This function takes in the matricies and index sets and comput the solution to the relaxed problem 
            to retrieve the stronly stationary point 
        
        Arg: 
                param as the output of computing_matricies()
                Iy, Iv: index sets from index_set(ỹ, ṽ)

        Return: 
                :d  : The load of datacenter
                :y  : The hidden neuron values
                :v  : Result of the ReLU function properity
                :obj: Sum of the final layer output 
                
        """
        c = param["c"]
        A = param["A"]
        f = param["f"]
        V = param["V"]
        W = param["W"]
        b = param["b"]
        ddim = n_in
        ydim = length(b)
        vdim = length(b)
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(model)
        @variable(model, d[1:ddim])
        @variable(model, y[1:ydim])
        @variable(model, v[1:vdim])
        @objective(model, Min, c'*y)
        @constraint(model, ineq, A*d .>= f)
        @constraint(model, eq, V*d .+ W*y .+ b .== v)

        @constraint(model, y[Iy] .== 0 )
        @constraint(model, v[Iy] .>= 0 )

        @constraint(model, y[Iv] .>= 0 )
        @constraint(model, v[Iv] .== 0 )
        optimize!(model)
        return Dict(:d=>JuMP.value.(d), :y=>JuMP.value.(y), :v=>JuMP.value.(v),:obj=>JuMP.objective_value(model), :status=>termination_status(model))
    end

    soln_RxLP = solve_RxLP(param,Iy,Iv)

    ȳ = soln_RxLP[:y]
    v̄ = soln_RxLP[:v]

    function multiplier_precondition(param,soln_RxLP)

        """

            This function takes in the parameter of matricies and the solution of RxLP 
            Return the set to indicate which indicies should be zero. This function serve
            to avoid the SOS1 constraint in the multiplier computation since which indicies should be zero
            is known to us

        Arg: 
                param: the matrix parameter calculated 
                solnRxLP: solution to RxLP
        Return: 
            Iλ: where λ is zero
            I0λ: where λ is not zero
            Iμy: where μy is zero
            Iμv: where μv is zero

      """
        ȳ = soln_RxLP[:y]
        v̄ = soln_RxLP[:v]
        # primer for ws solutions:
        λ_con = param["A"]*soln_RxLP[:d] - param["f"]
        Iλ = [] # this is the set where lambda has to be 0 
        I0λ = []# this is the set where lambda does not need to be 0 but still positive
        for i = 1:length(Iλ)

            global Iλ,I0λ

            if λ_con[i] != 0
                Iλ = [Iλ;i]
            else
                I0λ = [I0λ;i]
            end

        end


        Iμy = []# this is the set where μ has to be 0 
        Iμv = []
        for i = 1:length(ȳ)

            if ȳ[i] != 0
                Iμy = [Iμy;i]
            end

            if v̄[i] != 0
                Iμv = [Iμv;i]
            end

        end
        return Iλ,I0λ,Iμy,Iμv
    end

    Iλ,I0λ,Iμy,Iμv = multiplier_precondition(param,soln_RxLP)

    function solve_multipliers(param,Iλ,I0λ,Iμy,Iμv,Iv,Iy) 
       """
        This function takes in the parameter and the precondition sets and compute the 
        lagraugian multiplier of the RxLP solution to retrieve rhobar value

        Arg: 
            param: the matricies computed 
            Iλ,I0λ,Iμy,Iμv: index output of multiplier_precondition(param,soln_RxLP)
            Iv,Iy: output of  index_set(ỹ, ṽ)

        Return:
            A dictionary that contains 
                μy and μv for rho computation. 
       """
            
        c = param["c"]
        A = param["A"]
        V = param["V"]
        W = param["W"]
        b = param["b"]
        λdim = size(A,1)
        μydim = length(b)
        μvdim = size(V,1)
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        JuMP.set_silent(model)
        @variable(model, λ[1:λdim])
        @variable(model, μy[1:μydim])
        @variable(model, μv[1:μvdim])


        @constraint(model, eq1, A'*λ + V'*μv .== zeros(n_in,1))
        @constraint(model, eq2, W'*μv .+ μy .== c)
        
        @constraint(model, conλset, λ[Iλ] .== 0 )
        @constraint(model, conμyset, μy[Iμy] .== 0 )
        @constraint(model, conμwset, μv[Iμv] .== 0 )

        @constraint(model, con0λ, λ[I0λ] .>= 0 )
        @constraint(model, con0μy, μy[Iv] .>= 0 )
        @constraint(model, con0μw, μv[Iy] .>= 0 )
        
        optimize!(model)
        return Dict(:λ=>JuMP.value.(λ), :μy =>JuMP.value.(μy), :μv=>JuMP.value.(μv),:status=>termination_status(model))
    end 


    solved_multipliers = solve_multipliers(param,Iλ,I0λ,Iμy,Iμv,Iv,Iy) 

    function compute_ρ(solved_multipliers)

       """

        This function computes the \rho value from 
        the solution of the solution of 
        solve_multipliers(param,Iλ,I0λ,Iμy,Iμv,Iv,Iy) 

        Arg: 
            solved_multipliers: solution of solve_multipliers(param,Iλ,I0λ,Iμy,Iμv,Iv,Iy) 
        
        Return: \rho
        
        """

        μy = solved_multipliers[:μy]
        μv = solved_multipliers[:μv]
        ybar = ȳ[ȳ .> 0]
        μv = μv[ȳ .> 0]
        vbar = v̄[v̄ .> 0]
        μy = μy[v̄ .> 0]
        minρset = [0;-μv./ybar;-μy./vbar]
        ρ = maximum(minρset)
        return ρ
    end

    ρ = compute_ρ(solved_multipliers)

    return ρ
end

ρ  = rho_value()
