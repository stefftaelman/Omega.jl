module Prim

using ..Omega
using ..Omega: Ω, RandVar, URandVar, MaybeRV, ID, lift, uid, elemtype, isconstant
import ..Omega: params, name, ppapl, apl, reify
import Statistics: mean, var, quantile
import ..Causal: ReplaceRandVar
using Spec
import Distributions
const Djl = Distributions

export  bernoulli,
        betarv,
        β,
        categorical,
        # dirichlet,
        exponential,
        gammarv,
        Γ,
        invgamma,
        kumaraswamy,
        logistic,
        # mvnormal,
        normal,
        poisson,
        rademacher,
        uniform,

        mean

"Primitive random variable of known distribution"
abstract type PrimRandVar <: RandVar end  

"Name of a distribution"
function name end

name(t::T) where {T <: PrimRandVar} = T.name.name

"Parameters of `rv`"
@generated function params(rv::PrimRandVar)
  fields = [Expr(:., :rv, QuoteNode(f)) for f in fieldnames(rv) if f !== :id]
  Expr(:tuple, fields...)
end

ppapl(rv::PrimRandVar, ωπ) = rvtransform(rv)(ωπ, reify(ωπ, params(rv))...)

@generated function anysize(args::Union{<:AbstractArray, Real}...)
  isarr = (arg -> arg <: AbstractArray).([args...])
  firstarr = findfirst(isarr)
  if isempty(isarr)
    :(())
  else
    :(size(args[$firstarr]))
  end
end
@spec same(size.(filter(a -> a isa AbstractArray, args)))

include("univariate.jl")      # Univariate Distributions
include("multivariate.jl")    # Multivariate Distributions
include("statistics.jl")      # Distributional properties: mean, variance, etc
export  succprob,
        failprob,
        maximum,
        minimum,
        islowerbounded,                    
        isupperbounded,
        isbounded,
        std,
        median,
        mode,
        modes,

        skewness,
        kurtosis,
        isplatykurtic,
        ismesokurtic,

        isleptokurtic,
        entropy,
        mean
        
include("djl.jl")             # Distributions.jl interop

end