# NN_DC_OPT
Optimization over trained neural networks and application to data center scheduling

This repo produces the results for a small test case on PJM 5-bus and larger test case on IEEE 118 system of dca solution application on data center allocation.

Make sure to active project environment using `Project.toml` and `Manifest.toml` located in the folder.
To run this code, follow the instruction

1. Obtain the Gurobi licence
2. Download small_example or large_example folder
4. Open Terminal and cd to the corresponding folder

# Repo Method
4. Type `julia`, hit enter to enter julia environment
5. Type `]`
6. For small example, run this command
   ```
   activate .
   add JLD2 PowerModels Ipopt Distributions JuMP Gurobi LinearAlgebra Plots Random
   instantiate
     ```
   For large example, run this command
   ```
   activate .
   add JLD2 PowerModels Ipopt Distributions JuMP Gurobi LinearAlgebra Plots Random StatsBase Statistics
   instantiate
     ```
8. Click delete to go back to Julia environment
9. Type `include("dc_opt_dca.jl")` to run the file

# Repo Method
9. Type `exit()' to exit Julia
10. Type `julia dc_opt_dca.jl`, hit enter
# Output
You will obtain the plot named smallcase_convergence.png showing the convergence to groundtruth.

You will also obtain a dictionary dca_charges.jld2 containing the following 
1. ρ:                             ρ value used,
2. dca_charges:                   dca charges at each iteration,
3. ground_truth_charge_final:     sum of the ground truth charge,
4. dca_charges_final:             sum of the dca charge result,
5. sos1_cond:                     SOS1 condition violation,
6. iteration_number:              iteration_number.




# Package Specification

The code runs on the following version of the packages:

| Software   | Version 
|------------|-----------|
| Julia      | v1.8.5    |
| Gurobi     | 12.0      | 



**Gurobi is used as the optimization software. We use Gurobi with license version 12.0. A different version of Gurobi license will cause issue when running. If you have a different version please request a new academic license. Gurobi offers free academic licence [Gurobi](https://www.gurobi.com)**
JuMP is a modeling language. See [JuMP](https://github.com/jump-dev/JuMP.jl)
