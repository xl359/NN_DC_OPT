# NN_DC_OPT
Optimization over trained neural networks and application to data center scheduling

This repo produces the results for a small test case of dca solution application on data center allocation.

To run this code, follow the instruction

1. Obtain the Gurobi licence
2. Download everything to a folder
3. Open Terminal and cd to the corresponding folder

# Repo Method
4. Type julia, hit enter to enter julia environment
5. Type "]" then type "activate .", hit enter to active virtual environment
6. Type "instantiate", hit enter
7. Click delete to go back to julia environment
8. Type "include("dc_opt_dca.jl")" to run the file

# Repo Method
4. Type "julia dc_opt_dca.jl", hit enter


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

Gurobi offers free educational licence https://www.gurobi.com
