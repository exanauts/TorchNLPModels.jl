module TorchNLPModels

using NLPModels, PyCall, CUDA, DLPack
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
    obj_func::Function
    # tensor_cache::PyObject
    x::CuVector{T}
    py_x::PyObject
    dl_x::PyObject
    dx::CuVector{T}
    dl_dx::PyObject
    device::PyObject
end

# function TorchNLPModel(x0, py_file::String, py_obj::String, py_grad::String, py_vhp::Union{String, Nothing}=nothing)
function TorchNLPModel(x0, py_file::String, py_obj::String)
    dir = dirname(py_file)
    file = splitext(basename(py_file))[1]

    pushfirst!(PyVector(pyimport("sys")."path"), dir)
    py_module = pyimport(file)
    nvar = length(x0)
    meta = NLPModelMeta(nvar; x0=x0, ncon = 0)
    counters = Counters()
    obj_func = x -> (y = py_module[py_obj](x); GC.gc(true); y)
    device = py"torch.device"("cuda")
    x_cache = CuVector(x0)
    dl_x = DLPack.share(x_cache, py"torch.from_dlpack")
    py_x = py"torch.from_dlpack"(dl_x)
    py_x.requires_grad_()
    dx_cache = CuVector(x0)
    dl_dx = DLPack.share(dx_cache, py"torch.from_dlpack")
    return TorchNLPModel(
        meta, counters, x0, obj_func,
        x_cache, py_x, dl_x, dx_cache,
        dl_dx, device
    )
end

function NLPModels.obj(nlp::TorchNLPModel{T,S}, x::Array{T}) where {T,S}
    copyto!(nlp.x, x)
    result = nlp.obj_func(nlp.dl_x)
    py"torch.cuda.empty_cache"()
    return convert(Float32, result)
end

function NLPModels.grad!(nlp::TorchNLPModel{T,S}, x::Array{T}, dx::Array{T}) where {T,S}
    copyto!(nlp.x, x)
    py"""
    import torch
    with torch.no_grad():
        $(nlp).py_x.copy_($(nlp).dl_x)
    """
    py_y = nlp.obj_func(nlp.py_x)
    py_grad = py"torch.autograd.grad"(py_y, nlp.py_x, create_graph=false)[1]

    # Convert the result back to a Julia array
    grad = py"as_numpy"(py_grad)
    dx .= convert(Array{Float32}, grad)
    GC.gc(true)
    py"torch.cuda.empty_cache"()
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
    py_x = py"torch.tensor"(x, requires_grad=true).to(py"torch.float32")
    py_v = py"torch.tensor"(v).to(py"torch.float32")

    # Call the Python vhp function
    py_vhp = py"torch.autograd.functional.vhp"(nlp.obj_func, py_x, py_v)[2]

    # Convert the result back to a Julia array
    vhp = py"as_numpy"(py_vhp)
    Hv .= convert(Array{Float32}, vhp)
    return Hv
end

end