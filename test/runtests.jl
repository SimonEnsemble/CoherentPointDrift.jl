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
    X = collect(readdlm("osu.csv")')
    
    # rotate at a known angle
    θ = π / 5
    R_known = rotation_matrix2d(θ)
    Y = R_known * X
    
    # see if we can recover the rotation matrix
    R = affine_point_set_registration(X, Y)
    @test isapprox(R, rotation_matrix2d(-θ), atol=0.001) # well, need -θ rotation to rotate back.

    # a visual test; when we add noise
    Y = R_known * X .+ 0.01 * randn(size(X))
    
    # plot original points and points rotated, with noise.
    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y[1, :], y=Y[2, :], Geom.point, themes[2]),
             Guide.title("before alignment"), themes[1],
             Guide.manual_color_key("", ["reference", "mis-aligned reference"], [color_palette[1], color_palette[2]])
       )
    draw(PNG("before_alignment.png", 5inch, 4inch, dpi=300), p)

    R = affine_point_set_registration(X, Y, nb_em_steps=6)
    Y_transformed = R * Y

    p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, themes[1]),
             layer(x=Y_transformed[1, :], y=Y_transformed[2, :], Geom.point, themes[3]),
             Guide.title("after rotation"), themes[1],
             Guide.manual_color_key("", ["reference", "transformed"], [color_palette[1], color_palette[3]])
        )
    draw(PNG("after_alignment.png", 5inch, 4inch, dpi=300), p)
end
