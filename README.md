# NN_DC_OPT
Optimization over trained neural networks and application to data center scheduling

This repo produces the results for a small test case of dca solution application on data center allocation.

Make sure to active project environment using `Project.toml` and `Manifest.toml` located in the folder.
To run this code, follow the instruction

1. Obtain the Gurobi licence
2. Download everything to a folder
3. Open Terminal and cd to the corresponding folder

# Repo Method
4. Type `julia`, hit enter to enter julia environment
5. Type `]`
6. Run this command
   ```
   activate .
   add JLD2 PowerModels Ipopt Distributions JuMP Gurobi LinearAlgebra Plots Random
   instantiate
     ```
9. Click delete to go back to Julia environment
10. Type `include("dc_opt_dca.jl")` to run the file

# Repo Method
10. Type `exit()' to exit Julia
11. Type `julia dc_opt_dca.jl`, hit enter
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

BSON v0.3.9
CSV v0.10.15
CUDA v4.4.2
ColorTypes v0.12.0
Colors v0.13.0
DataFrames v1.7.0
Distributions v0.25.117
Flux v0.13.17
Gurobi v1.7.0
IJulia v1.26.0
Ipopt v1.7.2
JLD2 v0.5.11
JuMP v1.24.0
LaTeXStrings v1.4.0
MLDatasets v0.7.18
PlotlyJS v0.18.15
Plots v1.40.9
PowerModels v0.21.3
PrettyTables v2.3.2
ReverseDiff v1.15.3
StatsBase v0.34.4
StatsPlots v0.15.7
Turing v0.34.1
Zygote v0.6.75


**Gurobi is used as the optimization software. We use Gurobi with license version 12.0. A different version of Gurobi license will cause issue when running. If you have a different version please request a new academic license. Gurobi offers free academic licence [Gurobi](https://www.gurobi.com)**
JuMP is a modeling language. See [JuMP](https://github.com/jump-dev/JuMP.jl)
