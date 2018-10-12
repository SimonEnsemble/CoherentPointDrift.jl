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

@testset "2D visual test" begin
    # read in a 2D point cloud
    Y = collect(readdlm("osu.csv")')
    
    # rotate at a known angle
    θ = 2 * π * rand()
    R_known = rotation_matrix2d(θ)
    t_known = [rand(), rand()]

    # construct X as a known transformation of Y
    X = R_known * Y .+ t_known .+ 0.00001 * randn(size(Y))
    
    # see if we can recover the rotation matrix
    R, t, σ², q = affine_point_set_registration(X, Y, allow_translation=true, nb_em_steps=15)
    @test isapprox(R, R_known, atol=0.001) # well, need -θ rotation to rotate back.
    @test isapprox(t, t_known, atol=0.001)

    # a visual test; when we add noise
    X = R_known * Y .+ t_known .+ 0.01 * randn(size(X))
    
    # plot original points and points rotated, with noise.
    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y[1, :], y=Y[2, :], Geom.point, themes[2]),
             Guide.title("before alignment"), themes[1],
             Guide.manual_color_key("", ["reference", "mis-aligned reference"], [color_palette[1], color_palette[2]])
       )
    draw(PNG("before_alignment.png", 5inch, 4inch, dpi=300), p)

    R, t, σ², q = affine_point_set_registration(X, Y, nb_em_steps=8, allow_translation=true)
    Y_transformed = R * Y .+ t

    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y_transformed[1, :], y=Y_transformed[2, :], Geom.point, themes[3]),
             Guide.title("after rotation"), themes[1],
             Guide.manual_color_key("", ["reference", "transformed"], [color_palette[1], color_palette[3]])
        )
    draw(PNG("after_alignment.png", 5inch, 4inch, dpi=300), p)
end
