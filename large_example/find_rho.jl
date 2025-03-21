using JLD2
using Random
using StatsBase
using JuMP, Gurobi
using LinearAlgebra
const GRB_ENV = Gurobi.Env()

function rho_value()
    # we select which sample in the sample set to be the baselne
    sample_index = 28

    # we set the lower limit for ȳ and v̄ so that they are not too small to cause numerical error
    lim_rho = 0.008

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

    include("aux_fun_large.jl")

    net = dataset[:net]
    # load neural network
    nn = JLD2.load("nn_trained.jld2", "nn_trained")

    # nn size
    n_in, n_out = size(nn[:W][1],2), size(nn[:W][length(nn[:W])],1)
    n_hl, n_hn = length(nn[:W])-1, size(nn[:W][2],1)

    # function that let the input pass throught the neural network

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
        λ̂  = DNN[:W][3]*y5 + DNN[:b][3]
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

    ϕ(y,v) = 1/4*norm(y+v,2)^2 - 1/4*norm(y-v,2)^2
    f(λ̂,y,v) = ones(n_out)'_λ_(λ̂) + ρ*sum(ϕ(y[:,i],v[:,i]) for i in 1:n_hl)
    com_cond(y,v) = sum(y[:,i]'v[:,i] for i in 1:n_hl)

    # take the baseline distribution
    x̃ = d_test[:,sample_ind]
    base_d = d_test[:,sample_ind]

    Δ =  sum(x̃) # specify the summation constraint of the loads 

    # compute the upper limit

    _d̂_lower = 0.1*_d̂_(net[:dc])
    _d̂_upper = 1.9*_d̂_(net[:dc])

    # compute an initial solution
    sol_init = solve_init_6(nn,x̃);
    ỹ = sol_init[:y]
    ṽ = sol_init[:v]

    # if we have both y and v to be zero, its not allowed
    for i = 1:length(ỹ)
        if ỹ[i] == 0 && ṽ[i] == 0
            println("WARNING!!!")
        end
    end

    # given an upper and a lower bound of d and with the condition it has to sum to Δ, find many feasible
    # solutions

    function feasible_d_search(_d̂_lower,_d̂_upper,Δ,n_d)

        """

                    This function computes several feasible load value such that it satisfies the 
                    box constraint on the load and the summation of the load = Δ
                
                Arg: 
                        lower: lower bound on load
                        upper: upper bound on load
                        tot       : sum of the load constraint

                Return: 
                        :d  : load of datacenter. It returns n_d solutions
                        
                """

        function feasible_d(lower,upper,tot)
            """

                    This function computes several feasible load value such that it satisfies the 
                    box constraint on the load and the summation of the load = Δ
                
                Arg: 
                        lower: lower bound on load
                        upper: upper bound on load
                        tot       : sum of the load constraint

                Return: 
                        list of acceptable loads
                        
                """

            model = Model(() -> Gurobi.Optimizer(GRB_ENV))
            model = Model(optimizer_with_attributes(Gurobi.Optimizer, "PoolSearchMode"=>2, "PoolSolutions" => n_d, "PoolGap" => 500))
            JuMP.set_silent(model)
            @variable(model,d[1:n_in], Int)
            @constraint(model, d .>=lower)
            @constraint(model, d .<=  upper)
            @constraint(model, sum(d) == tot)
            optimize!(model)
            return Dict(:d => [value.(d; result = i) for i in 1:result_count(model)], :count => result_count(model))
        end

        # transform the constraints to integer space so that we can apply a cheap MIP to find 
        # several solution that is feasible to the constraints
        d_init = feasible_d(round.(_d̂_lower .*100) .+ 1,round.(_d̂_upper.*100) .- 1,round(Δ*100))
        # we transform the values back
        diff = (Δ*100 - round(Δ*100))/100
        for i = 1:n_d
            d_init[:d][i] = d_init[:d][i] ./ 100
            d_init[:d][i][1]  = d_init[:d][i][1] + diff
        end
        return d_init[:d]
    end
    d_initlist = feasible_d_search(_d̂_lower,_d̂_upper,Δ,n_d)




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



    
    function small_ρ_search()

        """
        This function go through the some feasible input d and find the 
        corresponding RxLP solution that generate ȳ and v̄ larger than the threshold

        """

        for i = 1:n_d
            global Iy, Iv, I0, soln_RxLP, ȳ, v̄, ỹ, ṽ, x̃
            x̃ = d_initlist[i]
            sol_init = solve_init_6(nn,x̃);
            ỹ = sol_init[:y]
            ṽ = sol_init[:v]
            for i = 1:length(ỹ)
                if ỹ[i] == 0 && ṽ[i] == 0
                    println("WARNING!!!")
                end
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

            # we solve RxLP to obtain the strongly stationary point
            soln_RxLP = solve_RxLP(param,Iy,Iv)
            ȳ = soln_RxLP[:y]
            v̄ = soln_RxLP[:v]
            
            y_filter = filter(x -> x != 0, ȳ)
            v_filter = filter(x -> x != 0, v̄)
            # we want the points to have y and v values to be larger than a threshold to avoid numerical error
            if all(x -> x > lim_rho, y_filter ) && all(x -> x > lim_rho, v_filter )
                break
            end
        end
        return soln_RxLP
    end
    soln_RxLP = small_ρ_search()
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

    ρ = compute_ρ(solved_multipliers) + 0.00001


    return ρ

end

ρ  = rho_value()
