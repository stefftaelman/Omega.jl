
"Soft Boolean.  Value in [o, 1]"
struct SoftBool{ET <: Real} <: AbstractSoftBool
  logerr::ET
  SoftBool(l::T) where {T} = new{T}(l)
  SoftBool(l::T) where {T <: ForwardDiff.Dual} = new{T}(l) # Resolves type ambiguity (from flux)
end
@invariant 0 <= err(b::SoftBool) <= 1

"Error in [0, 1]"
err(x::SoftBool) = exp(x.logerr)

"Log error"
logerr(x::SoftBool) = x.logerr
Bool(x::SoftBool) = logerr(x) == 0.0
ssofttrue() = SoftBool(0.0)
ssoftfalse() = SoftBool(-Inf)

## (In)Equalities

"Kernel return type as function of arguments"
kernelrettype(x::T, y::T) where T = T

"Soft Equality"
function ssofteq(x, y, k = globalkernel())
  r = d(x, y)
  SoftBool(k(r)::typeof(r))
end

"Soft >"
function ssoftgt(x::Real, y::Real, k = globalkernel())
  r = bound_loss(x, y, Inf)
  SoftBool(k(r)::typeof(r))
end

"Soft <"
function ssoftlt(x::Real, y::Real, k = globalkernel())
  r = bound_loss(x, -Inf, y)
  SoftBool(k(r)::typeof(r))
end

## Boolean Operators
## =================
function Base.:&(x::SoftBool, y::SoftBool)
  a = logerr(x)
  b = logerr(y)
  # c = min(a, b)
  c = a + b
  SoftBool(c)
end
Base.:|(x::SoftBool, y::SoftBool) = SoftBool(max(logerr(x), logerr(y)))
Base.all(xs::Vector{<:SoftBool}) = SoftBool(minimum(logerr.(xs)))
Base.all(xs::Vector{<:RandVar}) = RandVar(all, (xs, ))

# Arithmetic
Base.:*(x::SoftBool{T}, y::T) where T = SoftBool{T}(x.logerr * y)
Base.:*(x::T, y::SoftBool{T}) where T = SoftBool{T}(x * y.logerr)

## Show
## ====
Base.show(io::IO, sb::SoftBool) = print(io, "ϵ:$(logerr(sb))")