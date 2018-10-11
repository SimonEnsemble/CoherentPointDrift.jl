using LinearAlgebra
using DelimitedFiles
using Gadfly, Cairo

function initial_σ²(X::Array{Float64, 2}, Y::Array{Float64, 2})
    # nb of data pts, dimension
    D = size(X)[1]
    @assert D == size(Y)[1] "data must be of same dimension"
    M = size(Y)[2]
    N = size(X)[2]

    σ² = 0.0
    for n = 1:N
        for m = 1:M
            σ² += sum((X[:, n] - Y[:, m]) .^ 2)
        end
    end
    return σ² / (D * M  *N)
end

function affine_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2}; 
                                       nb_em_steps::Int=10)
    # nb of data pts, dimension
    D = size(X)[1]
    @assert D == size(Y)[1] "data must be of same dimension"
    M = size(Y)[2]
    N = size(X)[2]

    # the rotation matrix
    R = diagm(0 => [1.0 for i = 1:D])
    σ² = initial_σ²(X, Y) # variance in GMM
    P = zeros(M, N) # reuse

    for em_step = 1:nb_em_steps
        println("\tEM step: ", em_step)
        # conduct the rotation with our best guess of R
        Y_transformed = R * Y

        # E step. Get correspondence.
        for n = 1:N
            for m = 1:M
                P[m, n] = exp(-sum((Y_transformed[:, m] - X[:, n]) .^ 2) / (2 * σ²))
            end
            # at this pt row m is filled
            P[:, n] = P[:, n] / sum(P[:, n])
        end
        @assert isapprox(sum(P), N, atol=0.01)
        
        # M step. Get best transformation
        μx = X * transpose(P) * ones(M) / N
        μy = Y * P * ones(N) / N
        Xhat = broadcast(-, X, μx)
        Yhat = broadcast(-, Y, μy)

        A = Xhat * transpose(P) * transpose(Yhat)

        F = svd(A)
        
        R = F.U * diagm(0 => [i == D ? det(F.U * F.Vt) : 1 for i = 1:D]) * F.Vt

        σ² = (tr(Xhat * diagm(0 => transpose(P) * ones(M)) * transpose(Xhat)) - tr(transpose(A) * R)) / (N * D)
    end
    return R
end

# test
X = collect(readdlm("osu.csv")')
θ = π / 5
rotation_matrix2d(θ::Float64) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
R_known = rotation_matrix2d(θ)
Y = R_known * X .+ 0.01 * randn(size(X))

p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, Theme(default_color=colorant"green", background_color=colorant"white")),
         layer(x=Y[1, :], y=Y[2, :], Geom.point, Theme(default_color=colorant"blue", background_color=colorant"white")),
         Guide.title("b4 rotation")
   )
img = PNG("b4_alignment.png", 4inch, 4inch, dpi=300)
draw(img, p)

R = affine_point_set_registration(X, Y)
Y_transformed = R * Y

p = plot(layer(x=X[1, :], y=X[2, :], Geom.point, Theme(default_color=colorant"green", background_color=colorant"white")),
         layer(x=Y_transformed[1, :], y=Y_transformed[2, :], Geom.point, Theme(default_color=colorant"orange", background_color=colorant"white")),
         Guide.title("after rotation")
    )
img = PNG("after_alignment.png", 4inch, 4inch, dpi=300)
draw(img, p)
@assert isapprox(R, rotation_matrix2d(-θ), atol=0.01)
