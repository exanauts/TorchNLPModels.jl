using PyCall
using TorchNLPModels
using HDF5
using JSOSolvers

sys = pyimport("sys")
pushfirst!(PyVector(sys["path"]), "$(@__DIR__)/../../surrogate-downscaling/FourCastNet")
@pyinclude("$(@__DIR__)/../../surrogate-downscaling/bfgs.py")

torch_file = "$(@__DIR__)/../../surrogate-downscaling/bfgs.py"

x0 = [-1.2, 1.0]
nlp = TorchNLPModel(x0, torch_file, "J_func")
stats = lbfgs(nlp)