struct BFGS{T, VT <: AbstractVector{T}}
    init_strategy::BFGSInitStrategy
    is_instantiated::Base.RefValue{Bool}
    sk::VT
    yk::VT
    bsk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end

function create_quasi_newton(
    ::Type{BFGS},
    cb::AbstractCallback{T,VT},
    n;
    options=QuasiNewtonOptions(),
    ) where {T,VT}
    BFGS(
        options.init_strategy,
        Ref(false),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::BFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    yksk = dot(sk, yk)
    if yksk < T(1e-8)
        return false
    end
    # Initial approximation (Nocedal & Wright, p.143)
    if !qn.is_instantiated[]
        sksk = dot(sk, sk)
        Bk[diagind(Bk)] .= yksk ./ sksk
        qn.is_instantiated[] = true
    end
    # BFGS update
    mul!(qn.bsk, Bk, sk)
    alpha1 = one(T) / dot(sk, qn.bsk)
    alpha2 = one(T) / yksk
    _ger!(-alpha1, qn.bsk, qn.bsk, Bk)  # Bk = Bk - alpha1 * bsk * bsk'
    _ger!(alpha2, yk, yk, Bk)           # Bk = Bk + alpha2 * yk * yk'
    return true
end