module TorchNLPModels

using NLPModels, PyCall
export TorchNLPModel, jac_dense!

function __init__()
    # Import the Python module
    py"""
    # import mwe
    import torch
    import numpy as np  # Import numpy

    def as_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.asarray(tensor)
    """
end

struct TorchNLPModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    x0::S
    obj_func::PyObject
    # grad_func::PyObject
    # vhp_func::Union{PyObject, Nothing}
end

# function TorchNLPModel(x0, py_file::String, py_obj::String, py_grad::String, py_vhp::Union{String, Nothing}=nothing)
function TorchNLPModel(x0, py_file::String, py_obj::String)
    dir = dirname(py_file)
    file = basename(py_file)
    pushfirst!(PyVector(pyimport("sys")."path"), dir)
    py_module = pyimport(file)
    nvar = length(x0)
    meta = NLPModelMeta(nvar; ncon = 0)
    counters = Counters()
    obj_func = py_module[py_obj]
    # grad_func = py_module[py_grad]
    # vhp_func = isnothing(py_vhp) ? nothing : py_module[py_vhp]
    return TorchNLPModel(meta, counters, x0, obj_func)
end

function NLPModels.obj(nlp::TorchNLPModel{T,S}, x::Array{T}) where {T,S}
    # Convert Julia array to PyTorch tensor
    py_x = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")

    # Call the Python obj function
    result = nlp.obj_func(py_x)

    # Convert the result back to a Julia scalar
    return convert(Float64, result)
end

function NLPModels.grad!(nlp::TorchNLPModel{T,S}, x::Array{T}, dx::Array{T}) where {T,S}
    # Convert Julia array to PyTorch tensor
    py_x = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")

    # Call the Python obj function to compute the objective value
    py_y = nlp.obj_func(py_x)
    # higher_order = isnothing(nlp.vhp_func) ? false : true
    py_grad = py"torch.autograd.grad"(py_y, py_x, create_graph=true)[1]

    # Convert the result back to a Julia array
    grad = py"as_numpy"(py_grad)
    dx .= convert(Array{Float64}, grad)
    return dx
end

function NLPModels.hprod!(
    nlp::TorchNLPModel{T,S},
    x::Vector{T},
    v::Vector{T},
    Hv::Vector{T};
    obj_weight::T=one(T)
) where {T,S}
    # Convert Julia arrays to PyTorch tensors
    py_x = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")
    py_v = py"torch.tensor"(v).to(py"torch.float64")

    # Call the Python vhp function
    py_vhp = py"torch.autograd.functional.vhp"(nlp.obj_func, py_x, py_v)[2]

    # Convert the result back to a Julia array
    vhp = py"as_numpy"(py_vhp)
    Hv .= convert(Array{Float64}, vhp)
    return Hv
end

end