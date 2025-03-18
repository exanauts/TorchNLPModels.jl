module TorchNLPModels

using NLPModels

include("pycall.jl")
struct TorchModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    # obj::Function
    # grad::Function
    # hess_structure!::Function
    # hess_coord!::Function
    # hprod!::Function
    x0::S
end

function TorchModel(x0)
    nvar = length(x0)
    @show nvar
    meta = NLPModelMeta(nvar; ncon = 0)
    counters = Counters()
    return TorchModel(meta, counters, x0)
end

function NLPModels.obj(::TorchModel, x)
    return torch_obj(x)
end

function NLPModels.grad!(::TorchModel, x, dx)
    dx .= torch_grad(x)
end

# function NLPModels.hess_structure!(::TorchModel, hessian_structure)
#     fill!(hessian_structure, true)
# end

function NLPModels.hess_structure!(nlp::TorchModel, hrows::Vector{Int64}, hcols::Vector{Int64})
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
    ::TorchModel{T,S},
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector
) where {T,S}
    Hv .= torch_vhp(x, v)
end

function NLPModels.hess_coord!(
    nlp::TorchModel{T,S},
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
    nlp::TorchModel{T,S},
    x::AbstractVector{T},
    y::AbstractVector{T},
    hessian::AbstractVector{T}; kwargs...
) where {T,S}
    println("hess_coord2!")
    return NLPModels.hess_coord!(nlp, x, hessian)
end
end
