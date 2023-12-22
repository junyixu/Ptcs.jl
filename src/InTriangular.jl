module InTriangular
export is_inside
using OffsetArrays
mycross(a::Vector, b::Vector)= a[1]*b[2]- a[2]*b[1]
function sign_cross(x::Vector, a::Vector, b::Vector)
	mycross(a-x, a-b) |> sign
end
# x is inside (a,b,c) ?
is_inside(x::Vector,a::Vector,b::Vector,c::Vector)=(mycross(a-x, a-b) ≈ 0 || mycross(b-x, b-c) ≈ 0 || mycross(c-x, c-a) ≈ 0) || (sign_cross(x, a, b) == sign_cross(x, b, c) == sign_cross(x, c, a))  # 在三角形边界上也算作在三角形内部
is_inside(x::Vector,m::Matrix)=is_inside(x, m[:,1], m[:,2], m[:,3] )
end
