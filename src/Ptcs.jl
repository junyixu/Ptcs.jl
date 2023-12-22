#=
    Data analysis for APT
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

module Ptcs
using LinearAlgebra:⋅
using HDF5
using OffsetArrays
include("InTriangular.jl")
using .InTriangular
const mα=6.6446573357e-27
const me=9.1093837015e-31

import Base: size, length
export Unit, Particle, Particles, MetaData, is_in_flux_surface, get_psi_by_pos, get_vertex_id_by_pos, cart2cyld, is_dead, is_passage

struct Unit
    B::Float64
    E::Float64 # 电场
    Ω::Float64
    t::Float64
    x::Float64 # 位置
    p::Float64
    v::Float64
    ε::Float64 # 能量
    q::Float64
    m::Float64
    function Unit(B::Float64, q::Float64, m::Float64)
		c_0 = 2.99792458e8
        E = B*c_0;
        Ω=q*B/m;
        t= 1/Ω;
        x = m*c_0/(q*B);
        p=m*c_0;
        v=c_0;
        ε=m*c_0^2;
        new(B, E, Ω, t, x, p, v, ε, q, m)
    end
end

function Unit(B::Float64, type::String)
	if type == "alpha"
		return Unit(B, 2*1.60217733e-19, mα) # 有时候需要改单位电量和单位质量
	end
	return Unit(B, 1.60217733e-19, me)
end
Unit(B::Float64)=Unit(B, 1.60217733e-19, 9.1093837015e-31)

struct Particle
	dT::Float64
	die_step::Int
	isDead::Bool
	X::Matrix
	P::Matrix
	B::Matrix
	l::Int # 总步数
	function Particle(PTC::Matrix, B::Matrix, u::Unit)
		die = PTC[:, 1] .!= 0.0
		dT = PTC[2,2] * u.t
		die_step = 1
		isDead = false
		for i in eachindex(die) # 在第几步死亡
			if die[i] == 1.0
				die_step = i
				isDead = true
				break
			end
		end
		if !isDead
			die_step = length(die)+1
		end
		X = PTC[:, 3:5] * u.x
		P = PTC[:, 6:8] * u.p
		new(dT, die_step, isDead, X, P, B*u.B, size(PTC, 1))
	end
end

function Particle(filename::String, u::Unit)
	fid=h5open(filename, "r")
	PTC = read(fid["PTC"])
	B = read(fid["B"])
	close(fid)
	return Particle(PTC, B, u)
end

# 初始 pitch angle
function cos(P::Matrix, B::Matrix)::Vector # P: 3xn B: 3xn
	# 初始磁场
	B_norm = sqrt.(sum(B.^2, dims=1))
	# 初始动量
	p_norm = sqrt.(sum(P.^2, dims=1))
	# 初始 pitch angle
	return sum(P.*B, dims=1) ./ B_norm ./ p_norm |> vec
end

function cos(p::Vector, B::Vector)::AbstractFloat
	B_norm = (sqrt ∘ sum)(B.^2)
	p_norm = (sqrt ∘ sum)(p.^2)
	return sum(p.*B) / B_norm / p_norm
end

# %%
struct Particles
	ptcs::Vector{Particle}
	dT::Float64 # 步长/秒 
	X0::Matrix # 初始位置/米
	P0::Matrix # 初始动量
	B0::Matrix # 初始磁场/特斯拉
	die::Vector # 在第i步死亡的粒子数
	cosθ::Vector 
	n::Int # 粒子数
	l::Int # 总步数
	function Particles(filename::String, u::Unit)
		fid=h5open(filename, "r")
		# for key in keys(fid) # 读取全部数据
		# 	sb=Symbol(key)
		# 	eval(:($sb = read(fid[$key])))
		# end
		PTC = read(fid["PTC"])
		B = read(fid["B"])
		close(fid)

		dT = PTC[2,2] * u.t

		n = Int(size(PTC, 2)/11) # size, n ptcs
		l = size(PTC, 1) # size, n ptcs

		ptcs = Vector{Particle}(undef, n)
		for i in 1:Int(size(PTC, 2)/11)
			ptcs[i] = Particle(PTC[:, (i-1)*11+1:i*11], B[:, (i-1)*3+1:i*3], u)
		end

		die = zeros(Int, l)
		for i in 1:n
			if ptcs[i].isDead
				die[ptcs[i].die_step]+=1
			end
		end

		X0 = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.X[1, :])
		P0 = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.P[1, :])
		B0 = reshape(B[1, :], 3, :) * u.B
		new(ptcs, dT, X0, P0, B0, die, cos(P0, B0), n, l)
	end
	function Particles(ptcs::Vector{Particle})
		n = length(ptcs)
		l = ptcs[1].l
		die = zeros(Int, l)
		for i in 1:n
			if ptcs[i].isDead
				die[ptcs[i].die_step]+=1
			end
		end
		X0 = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.X[1, :])
		P0 = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.P[1, :])
		B0 = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.B[1, :])
		dT = ptcs[1].dT
		new(ptcs, dT, X0, P0, B0, die, cos(P0, B0), n, l)
	end
end
# %%

Base.length(ptc::Particle) = size(ptc.X , 1)
Base.size(ptc::Particle) = (size(ptc.X , 1),)
Base.length(ptcs::Particles) = length(ptcs.ptcs[1]) # 总时间步数目
Base.size(ptcs::Particles) = size(ptcs.ptcs, 1)

x(ptc::Particle)::Vector = ptc.X[:, 1]
y(ptc::Particle)::Vector = ptc.X[:, 2]
z(ptc::Particle)::Vector = ptc.X[:, 3]
R(ptc::Particle)::Vector = sqrt.(x(ptc).^2 + y(ptc).^2)
function R(X::Matrix)
	if size(X, 2) != 3
		error("不是 (x,y,z) 矩阵")
	end
	@. sqrt(X[:, 1]^2 + X[:, 2])
end
function z(X::Matrix)
	if size(X, 2) != 3
		error("不是 (x,y,z) 矩阵")
	end
	X[:, 3]
end
px(ptc::Particle)::Vector = ptc.P[:, 1]
py(ptc::Particle)::Vector = ptc.P[:, 2]
pz(ptc::Particle)::Vector = ptc.P[:, 3]
p_norm(ptc::Particle)::Vector = sqrt.(px(ptc).^2 + py(ptc).^2 + pz(ptc).^2)

x(ptcs::Particles)::Matrix = hcat([x(ptcs.ptcs[i]) for i in 1:size(ptcs)]...)
y(ptcs::Particles)::Matrix = hcat([y(ptcs.ptcs[i]) for i in 1:size(ptcs)]...)
z(ptcs::Particles)::Matrix = hcat([z(ptcs.ptcs[i]) for i in 1:size(ptcs)]...)
R(ptcs::Particles)::Matrix = hcat([R(ptcs.ptcs[i]) for i in 1:size(ptcs)]...)
z(ptcs::Vector{Particle})::Matrix = hcat([z(ptcs[i]) for i in 1:length(ptcs)]...)
R(ptcs::Vector{Particle})::Matrix = hcat([R(ptcs[i]) for i in 1:length(ptcs)]...)

X(ptcs::Particles, i::Int) = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.X[i, :])
P(ptcs::Particles, i::Int) = reduce(hcat, [ptcs[i] for i in 1:n] .|> ptc->ptc.P[i, :])

is_dead(p::Particle)::Bool = p.isDead
is_dead(p::Particles)::BitVector = [is_dead(ptc) for ptc in p.ptcs]

struct MetaData
	T0::OffsetArray{Int}
	Tensor::OffsetArray{Int}
	T1::OffsetArray{Int}
	T2::OffsetArray{Int}
	XY::OffsetArray{Float64}
	d::Vector{Float64}
	o::Vector{Float64}
	s::Vector{Float64}
	function MetaData(filename::String)
		h5open(filename, "r") do fid
			T0= read(fid["T0/data"])
			Tensor = similar(T0, reverse(size(T0)))
			for i = 1:size(T0, 3)
				Tensor[i, :, :] = T0[:, :, i]'
			end
			Tensor=OffsetArray(Tensor, -1, -1, -1)
			T0=OffsetArray(T0, -1, -1, -1)
			T1=OffsetArray(read(fid["T1/data"]), -1, -1)
			T2=OffsetArray(read(fid["T2/data"]), -1, -1)
			XY=OffsetArray(read(fid["XY/data"]), -1, -1)
			new(T0, Tensor, T1, T2, XY,read(fid["d"]), read(fid["o"]), read(fid["s"]))
		end
	end
end


"""
通过 vertex 编号知道该 vertex 所对应的磁面 ψ 是第几层，
其中 20201 是 XY 总的点的个数，4 是公差
"""
i2ψ(i::Int)::Int=(i-1)%Int((20201 -1)/4) |> y -> 0.5*(-1 + sqrt(1+8y)) |> x->floor(Int, x)

"""
判断点的坐标是否在最外层磁面内

	is_in_flux_surface(xy::AbstractVector, md::MetaData)

"""
function is_in_flux_surface(xy::AbstractVector, MD::MetaData) # 出 ITER 非截面
	i,j= floor.(Int, (parent(xy) .- MD.o)./MD.d)
	if 0<i<MD.s[1] && 0<j<MD.s[2]
		for vertex in MD.T0[i, j, 1:MD.T0[i, j, 0]]
			for tri in MD.T1[0:MD.T1[6, vertex]-1, vertex]
				if is_inside(parent(xy), parent(MD.XY[:, MD.T2[:, tri]]))
					return true
				end
			end
		end
	end
	return false
end

"""
通过点的坐标找到周围的数据点的编号

# Example

```julia-repl
julia> get_vertex_id_by_pos([r-R0,z]./u.x, MD)
3-element Vector{Int64}:
 14720
 14721
 14817
```
"""
function get_vertex_id_by_pos(xy::AbstractVector, MD::MetaData)::Vector{Int}
	i,j= floor.(Int, (parent(xy) .- MD.o)./MD.d)
	if 0<i<MD.s[1] && 0<j<MD.s[2]
		for vertex in MD.T0[i, j, 1:MD.T0[i, j, 0]]
			for tri in MD.T1[0:MD.T1[6, vertex]-1, vertex]
				if is_inside(parent(xy), parent(MD.XY[:, MD.T2[:, tri]]))
					return parent(MD.T2[:, tri])
				end
			end
		end
	end
	error("出界!!!")
	return [-1,-1,-1]
end


function get_vertex_id_by_pos_new(xy::AbstractVector, MD::MetaData)::Vector{Int}
	i,j= floor.(Int, (parent(xy) .- MD.o)./MD.d)
	if 0<i<MD.s[1] && 0<j<MD.s[2]
		for vertex in MD.Tensor[1:MD.Tensor[0, j, i], j, i]
			for tri in MD.T1[0:MD.T1[6, vertex]-1, vertex]
				if is_inside(parent(xy), parent(MD.XY[:, MD.T2[:, tri]]))
					return parent(MD.T2[:, tri])
				end
			end
		end
	end
	error("出界!!!")
	return [-1,-1,-1]
end


"""
通过点的坐标判断该点在磁面第几层，
层数从 1 开始数，即，最小层数为 1
"""
function get_psi_by_pos(xy::AbstractVector, MD::MetaData)::Int # 出 ITER 非截面
	return get_vertex_id_by_pos(xy, MD) .|> i2ψ |> maximum
end

function get_psi_by_pos(XY::Matrix, MD::MetaData) # 出 ITER 非截面
	ψs = zeros(size(XY, 2))
	for i in 1:size(XY, 2)
		ψs[i] = get_psi_by_pos(XY[:, i], MD::MetaData)
	end
	return ψs/100 # TODO 这里若 199 层，就 / 199
end
# %%

"""
Data.h5 大小: 多少 G
"""
data_size(p::Particles)=p.l*p.n*8*(11+3)/1024^3

"""
笛卡尔坐标转为柱坐标
"""
function cart2cyld(x::T,y::T,z::T) where T<:Real
	R = sqrt(x^2 + y^2);
	if R == 0
		ϕ = 0
		println("Singular point occors when CART to CYLD")
	end
	if y >= 0 
		ϕ = acos(x/R) 
	else
		ϕ = 2π-acos(x/R)
	end
	return [R, ϕ, z]
end
cart2cyld(v::Vector)=length(v) == 3 ? cart2cyld(v...) : error("dim != 3")

function cart2cyld(X::Matrix)
	rϕz = similar(X)
	if size(X,2) == 3
		for i = 1:size(X,1)
			rϕz[i, :] = cart2cyld(X[i, :])
		end
	else
		error("size 不为 3!!!")
	end
	return rϕz
end

function plot_tri(x::Vector,v::Matrix)
	plt.scatter(v[1, 1], v[2, 1], c="r")
	plt.scatter(v[1, 2], v[2, 2], c="g")
	plt.scatter(v[1, 3], v[2, 3], c="b")
	plt.plot(v[1, 1:2], v[2, 1:2])
	plt.plot(v[1, 2:3], v[2, 2:3])
	plt.plot(v[1, [3, 1]], v[2, [3,1]])
	plt.scatter(x[1], x[2], c="k")
	plt.show()
end


interpolateB(w::Vector{Float64}, subB::AbstractArray, id::Vector{Int})=sum(parent(subB[:, id]).*w', dims=2) |> vec

function cart2cyld(p::Particle)
	rϕz = similar(p.X)
	for i in 1:size(p.X, 1)
		rϕz[i, :] = cart2cyld(p.X[i, :])
	end
	return rϕz
end

"""
计算环向角
"""
function countϕ(ϕ::Vector{Float64})
	count = 0
	for i in 2:length(ϕ)
		if ϕ[i] < ϕ[i-1]
			count += 1
		end
	end
	return count
end
function countϕ(p::Particle)
	rϕz=cart2cyld(p)
	countϕ(rϕz[:, 2])
end


# %% gyro center {{{

"""
方均根
"""
rsq(B::AbstractMatrix) = sum(B.^2, dims=2) .|> sqrt |> vec

"""
用回旋半径
计算导心
``R = mv/qB``
``R = mv \\times B / (qB^2)``
	R = p × B / (u.q * sum(B.^2))
"""
function gyro_R(p::Particle)::Matrix
	R = similar(p.P)
	for i = 1:size(R, 1)
		P = p.P[i, :]
		B = p.B[i, :]
		R[i, :] = P × B / (u.q * sum(B.^2))
	end
	return R
end

function center(p::Particle)::Matrix
	p.X + gyro_R(p)
end


"""
是否是通行粒子
"""
is_passage(p::Particle)::Bool = p|> p->eachrow(p.B) .⋅ eachrow(p.P) .|> sign |> unique |> length == 1

#}}}


end
