module CoherentPointDrift

using LinearAlgebra
using DelimitedFiles

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

# eqn. (7) in paper
function q_objective(X::Array{Float64, 2}, Y::Array{Float64, 2}, P::Array{Float64, 2},
                     σ²::Float64, R::Array{Float64, 2}, t::Array{Float64, 1})
    D = size(X)[1]
    M = size(Y)[2]                                                                          
    N = size(X)[2]

    Y_transformed = R * Y .+ t

    q = 0.0
    for n = 1:N
        for m = 1:M
            q += P[m, n] * sum((X[:, n] - Y_transformed[:, m]) .^ 2)
        end
    end
    q /= 2 * σ²
    q += N * D / 2 * log(σ²)
    return q
end

"What transformation R * Y + t needs to be applied to Y to make it match X?"
function affine_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2};
                                       allow_translation::Bool=false,
                                       nb_em_steps::Int=5)
    # nb of data pts, dimension
    D = size(X)[1]
    @assert D == size(Y)[1] "data must be of same dimension"
    M = size(Y)[2]
    N = size(X)[2]

    # the rotation matrix
    R = diagm(0 => [1.0 for i = 1:D])
    # the translation vector
    t = zeros(D)
    # the bandwidth (variance) of the Gaussian mixture model
    σ² = initial_σ²(X, Y)
    # probabilities of correspondence
    P = rand(Float64, M, N)
    q = 0.0

    for em_step = 1:nb_em_steps
        println("\tEM step: ", em_step)
        # conduct the rotation with our best guess of R
        Y_transformed = R * Y .+ t

        # E step. Get correspondence.
        for n = 1:N
            for m = 1:M
                P[m, n] = exp(-sum((Y_transformed[:, m] - X[:, n]) .^ 2) / (2 * σ²))
            end
            # at this pt row m is filled
            P[:, n] = P[:, n] / sum(P[:, n])
        end
        @assert isapprox(sum(P), N, atol=0.05)
        
        # M step. Get best transformation
        μx = X * transpose(P) * ones(M) / N
        μy = Y * P * ones(N) / N
        Xhat = broadcast(-, X, μx)
        Yhat = broadcast(-, Y, μy)

        A = Xhat * transpose(P) * transpose(Yhat)

        F = svd(A)
        
        R = F.U * diagm(0 => [i == D ? det(F.U * F.Vt) : 1.0 for i = 1:D]) * F.Vt

        σ² = (tr(Xhat * diagm(0 => transpose(P) * ones(M)) * transpose(Xhat)) - tr(transpose(A) * R)) / (N * D)

        # objective
        q = q_objective(X, Y, P, σ², R, t)
        println("\t\tobjective: ", q)

        if allow_translation
            t = μx .- R * μy
        end
    end

    println("σ = ", sqrt(σ²))

    return R, t, σ², q
end

export affine_point_set_registration

end
