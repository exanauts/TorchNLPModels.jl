module MadNLPExt

using TorchNLPModels
using MadNLP

function MadNLP.jac_dense!(nlp::TorchNLPModel{T,S}, x, J) where {T,S}
    @assert size(J,1) == 0
    return nothing
end

end