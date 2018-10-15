using LinearAlgebra
using DelimitedFiles
using Gadfly, Cairo, Colors
using Test

using CoherentPointDrift

# 2D rotation matrix for testing.   
rotation_matrix2d(θ::Float64) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

# pretty colors
color_palette = Scale.color_discrete().f(7)[[1, 4, 6]]
themes = [Theme(default_color=color, panel_fill="white", grid_color="lightgray", 
                background_color=parse(Colorant, "white")) for color in color_palette]

 # # to test arrays of different sizes and find absurd bugs
 # rigid_point_set_registration(rand(3, 10), rand(3, 15), allow_translation=true, nb_em_steps=4, w=0.2)
 # rigid_point_set_registration(rand(3, 15), rand(3, 10), allow_translation=true, nb_em_steps=4, w=0.2)

function test_transform(Y::Array{Float64, 2}, θ::Float64, t::Array{Float64, 1}, noise_level::Float64)
    R = rotation_matrix2d(θ)
    return R * Y .+ t .+ noise_level * randn(size(Y))
end

@testset "2D visual test" begin
    # read in a 2D point cloud
    Y = collect(readdlm("osu.csv")')
 #     Y = collect(readdlm("tree.csv")')
    
    ###
    #  rotate at a known angle, recover rotation
    ###
    θ = π / 6.0
    R_known = rotation_matrix2d(θ)
    X = test_transform(Y, θ, [0.0, 0.0], 0.0)
    R, t, σ², q = rigid_point_set_registration(X, Y, allow_translation=false, nb_em_steps=10, w=0.0)
    @test isapprox(R, R_known, atol=0.001)

    ###
    #  translate and rotate
    ###
    t_known = [0.25, 0.45]
    X = test_transform(Y, θ, t_known, 0.0)
    # see if we can recover the rotation matrix
    R, t, σ², q = rigid_point_set_registration(X, Y, allow_translation=true, nb_em_steps=15, w=0.0)
    @test isapprox(R, R_known, atol=0.001) # well, need -θ rotation to rotate back.
    @test isapprox(t, t_known, atol=0.001)
    
    ###
    #   now with some noise (and plot)
    ###
    X = test_transform(Y, θ, t_known, 0.005)

    # plot original points and points rotated, with noise.
    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y[1, :], y=Y[2, :], Geom.point, themes[2]),
             Guide.title("before alignment"), themes[1],
             Guide.manual_color_key("", ["reference", "mis-aligned reference"], [color_palette[1], color_palette[2]])
       )
    draw(PNG("before_alignment.png", 5inch, 4inch, dpi=300), p)

    R, t, σ², q = rigid_point_set_registration(X, Y, allow_translation=true, nb_em_steps=15, w=0.0)
    Y_transformed = R * Y .+ t

    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y_transformed[1, :], y=Y_transformed[2, :], Geom.point, themes[3]),
             Guide.title("after rotation"), themes[1],
             Guide.manual_color_key("", ["reference", "transformed"], [color_palette[1], color_palette[3]])
        )
    draw(PNG("after_alignment.png", 5inch, 4inch, dpi=300), p)
end
