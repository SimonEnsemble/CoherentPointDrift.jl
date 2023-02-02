using LinearAlgebra
using DelimitedFiles
using Test

using CoherentPointDrift

# 2D rotation matrix for testing.   
rotation_matrix2d(θ::Float64) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

function test_transform(Y::Array{Float64, 2}, θ::Float64, t::Array{Float64, 1}, noise_level::Float64)
    R = rotation_matrix2d(θ)
    return R * Y .+ t .+ noise_level * randn(size(Y))
end

@testset "2D visual test" begin
    # read in a 2D point cloud
    Y = collect(readdlm("osu.csv")')
    
    ###
    #  rotate at a known angle, recover rotation
    ###
    θ = π / 6.0
    R_known = rotation_matrix2d(θ)
    X = test_transform(Y, θ, [0.0, 0.0], 0.0)
    R, t, σ², q = rigid_point_set_registration(X, Y, allow_translation=false, w=0.0, max_nb_em_steps=35)
    @test isapprox(R, R_known, atol=0.01)

    ###
    #  translate and rotate
    ###
    t_known = [0.25, 0.45]
    X = test_transform(Y, θ, t_known, 0.0)
    # see if we can recover the rotation matrix
    R, t, σ², q = rigid_point_set_registration(X, Y, allow_translation=true, w=0.0, max_nb_em_steps=35)
    @test isapprox(R, R_known, atol=0.01) # well, need -θ rotation to rotate back.
    @test isapprox(t, t_known, atol=0.01)
end
