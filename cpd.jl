using LinearAlgebra

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
                                       nb_em_steps::Int=25)
    # nb of data pts, dimension
    D = size(X)[1]
    @assert D == size(Y)[1] "data must be of same dimension"
    M = size(Y)[2]
    N = size(X)[2]

    # the rotation matrix
    R = diagm(0 => [1.0, 1.0, 1.0])
    σ² = initial_σ²(X, Y) # variance in GMM
    P = zeros(M, N) # reuse

    for em_step = 1:nb_em_steps
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

        R = F.V * diagm(0 => [1, 1, det(F.U * F.Vt)]) * transpose(F.U)

        σ² = (tr(Xhat * diagm(0 => transpose(P) * ones(M)) * transpose(Xhat)) - tr(transpose(A) * transpose(R))) / (N * D)
    end
    return R
end

X = randn(3, 15)
Y = randn(3, 10)
affine_point_set_registration(X, Y)
