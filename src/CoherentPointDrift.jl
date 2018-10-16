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
    Np = sum(P)

    Y_transformed = R * Y .+ t

    q = 0.0
    for n = 1:N
        for m = 1:M
            q += P[m, n] * sum((X[:, n] - Y_transformed[:, m]) .^ 2)
        end
    end
    q /= 2 * σ²
    q += Np * D / 2 * log(σ²)
    return q
end

"What transformation R * Y + t needs to be applied to Y to make it match X?"
function rigid_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2};
                                      allow_translation::Bool=false,
                                      max_nb_em_steps::Int=25, q_tol::Float64=1e-5,
                                      w::Float64=0.0, verbose::Bool=true, σ²_tol::Float64=1e-14)
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
    P = zeros(Float64, M, N)
    q = 0.0
    q_old = Inf

    for em_step = 1:max_nb_em_steps
        # conduct the rotation with our best guess of R
        Y_transformed = R * Y .+ t

        # E step. Get correspondence.
        for n = 1:N
            for m = 1:M
                P[m, n] = exp(-sum((Y_transformed[:, m] - X[:, n]) .^ 2) / (2 * σ²))
            end
            # at this pt row m is filled
            P[:, n] = P[:, n] / (sum(P[:, n]) + (2*π*σ²)^(D/2) * w / (1.0 - w) * M / N)
        end
        Np = sum(P)
        
        # M step. Get best transformation
 #         μx = (sum(X * transpose(P), dims=2) / Np)[:]
 #         μy = (sum(Y * P, dims=2) / Np)[:]
        μx = X * P' * ones(M) / Np
        μy = Y * P * ones(N) / Np
        Xhat = broadcast(-, X, μx)
        Yhat = broadcast(-, Y, μy)

        A = Xhat * P' * Yhat'

        F = svd(A, full=true)
        
        # update transformation
        R = F.U * diagm(0 => [i == D ? det(F.U * F.Vt) : 1.0 for i = 1:D]) * F.Vt

        if allow_translation
            t = μx - R * μy
        end
        
        # update variance
        σ² = (tr(Xhat * diagm(0 => P' * ones(M)) * Xhat') - tr(A' * R)) / (Np * D)
        if σ² < 0.0
            σ² = 1e-6
        end

        # objective
        q = q_objective(X, Y, P, σ², R, t)
        
        # terminate if objective hasn't decreased much, suggesting convergence
        if abs(q - q_old) < q_tol || (σ² < σ²_tol)
            break
        end

        if verbose
            println("\tEM step: ", em_step)
            println("\t\tobjective: ", q)
            println("\t\tσ² = ", σ²)
        end
        
        q_old = q
    end

    return R, t, σ², q
end

export rigid_point_set_registration

end
