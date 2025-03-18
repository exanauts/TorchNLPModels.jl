module TorchNLPModels

using NLPModels, PyCall
export TorchNLPModel

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

# include("pycall.jl")
struct TorchNLPModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    x0::S
    obj_func::PyObject
    grad_func::PyObject
    vhp_func::Union{PyObject, Nothing}
end

function TorchNLPModel(x0, py_file::String, py_obj::String, py_grad::String, py_vhp::Union{String, Nothing}=nothing)
    dir = dirname(py_file)
    file = basename(py_file)
    pushfirst!(PyVector(pyimport("sys")."path"), dir)
    py_module = pyimport(file)
    nvar = length(x0)
    meta = NLPModelMeta(nvar; ncon = 0)
    counters = Counters()
    obj_func = py_module[py_obj]
    grad_func = py_module[py_grad]
    vhp_func = isnothing(py_vhp) ? nothing : py_module[py_vhp]
    return TorchNLPModel(meta, counters, x0, obj_func, grad_func, vhp_func)
end

function NLPModels.obj(nlp::TorchNLPModel{T,S}, x::Array{T}) where {T,S}
    # Convert Julia array to PyTorch tensor
    input_tensor_torch = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")

    # Call the Python obj function
    result = nlp.obj_func(input_tensor_torch)

    # Convert the result back to a Julia scalar
    return convert(Float64, result)
end

function NLPModels.grad!(nlp::TorchNLPModel{T,S}, x::Array{T}, dx::Array{T}) where {T,S}
    # Convert Julia array to PyTorch tensor
    input_tensor_torch = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")

    # Call the Python grad function
    # result = py"mwe.grad"(py"mwe.f", input_tensor_torch)
    result = nlp.grad_func(nlp.obj_func, input_tensor_torch)

    # Convert the result back to a Julia array
    numpy_result = py"as_numpy"(result)
    dx .= convert(Array{Float64}, numpy_result)
    return dx
end

# function NLPModels.hess_structure!(::TorchNLPModel, hessian_structure)
#     fill!(hessian_structure, true)
# end

function NLPModels.hess_structure!(nlp::TorchNLPModel, hrows::Vector{Int64}, hcols::Vector{Int64})
    m = length(nlp.x0)
    n = length(nlp.x0)
    k = 1
    for i in 1:m
            for j in 1:n
                if i <= j
                    hrows[k] = i
                    hcols[k] = j
                    k += 1
                end
            end
    end
end

function NLPModels.hprod!(
    nlp::TorchNLPModel{T,S},
    x::Vector{T},
    v::Vector{T},
    Hv::Vector{T};
    obj_weight::T=one(T)
) where {T,S}
    # Convert Julia arrays to PyTorch tensors
    input_tensor_torch = py"torch.tensor"(x, requires_grad=true).to(py"torch.float64")
    v_vector_torch = py"torch.tensor"(v).to(py"torch.float64")

    # Call the Python vhp function
    result = nlp.vhp_func(nlp.obj_func, input_tensor_torch, v_vector_torch)

    # Convert the result back to a Julia array
    numpy_result = py"as_numpy"(result)
    Hv .= convert(Array{Float64}, numpy_result)
    return Hv
end

function NLPModels.hess_coord!(
    nlp::TorchNLPModel{T,S},
    x::AbstractVector{T},
    hessian::AbstractVector{T}; kwargs...
) where {T,S}
    println("hess_coord!")
    n = length(x)
    v = zeros(n)
    H = zeros(n, n)
    for i in 1:n
        v[i] = 1.0
        hprod!(nlp, x, v, H[i, :])
        v[i] = 0.0
    end
    k = 1
    for i in 1:n
            for j in 1:n
                if i <= j
                    hessian[k] = H[i,j]
                    k += 1
                end
            end
    end
    return hessian
end

function NLPModels.hess_coord!(
    nlp::TorchNLPModel{T,S},
    x::AbstractVector{T},
    y::AbstractVector{T},
    hessian::AbstractVector{T}; kwargs...
) where {T,S}
    println("hess_coord2!")
    return NLPModels.hess_coord!(nlp, x, hessian)
end
end
