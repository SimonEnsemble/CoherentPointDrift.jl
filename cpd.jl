using LinearAlgebra

function initial_σ²(X::Array{Float64, 2}, Y::Array{Float64, 2})
    @assert size(X)[2] == size(Y)[2] "data must be of same dimension"
    σ² = 0.0
    for i = 1:size(X)[1]
        for j = 1:size(Y)[1]
            # TODO make this column-major...
            σ² += sum((X[i, :] - Y[j, :]) .^ 2)
        end
    end
    return σ² / (size(X)[2] * size(X)[1] * size(Y)[1])
end

function affine_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2})
    # number of data pts in each pt. cloud
    M = size(Y)[1]
    N = size(X)[1]
    # dimension
    D = size(X)[2]
    @assert D == size(Y)[2] "data must be same dimension"

    # initialize
    w = 0 # don't use but implicitly here I'm assuming no noise.
    B = diagm(0 => [1.0, 1.0, 1.0])
    σ² = initial_σ²(X, Y) # variance in GMM
    P = zeros(M, N) # reuse

    for em_step = 1:15
        # E step. Get correspondence.
        for m = 1:M
            for n = 1:N
                y_m_transformed = B * Y[m, :]
                P[m, n] = exp(-sum((y_m_transformed - X[n, :]) .^ 2) / (2 * σ²))
            end
            # at this pt row m is filled
        end
        for n = 1:N
            P[:, n] = P[:, n] / sum(P[:, n])
        end
        @assert isapprox(sum(P), N, atol=0.01)
        
        # M step. Get best transformation
        μx = transpose(X) * transpose(P) * ones(M) / N
        μy = transpose(Y) * P * ones(N) / N
        Xhat = X - ones(N) * transpose(μx)
        Yhat = Y - ones(M) * transpose(μy)
        B = transpose(Xhat) * transpose(P) * Yhat * inv(transpose(Yhat) * diagm(0 => P * ones(size(P)[2])) * Yhat)
        σ² = (tr(transpose(Xhat) * diagm(0 => transpose(P) * ones(size(P)[1])) * Xhat) - tr(transpose(Xhat) * transpose(P) * Yhat * transpose(B))) / D / N
    end
    return B
end

affine_point_set_registration(rand(15, 3), rand(10, 3))
