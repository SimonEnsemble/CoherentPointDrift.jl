module CoherentPointDrift

using LinearAlgebra
# using DelimitedFiles
using Printf

function dmn(X::Array{Float64, 2}, Y::Array{Float64, 2})
    # nb of data pts, dimension
    D = size(X)[1]
    @assert D == size(Y)[1] "data must be of same dimension"
    M = size(Y)[2]
    N = size(X)[2]
    return D, M, N
end

function initial_σ²(X::Array{Float64, 2}, Y::Array{Float64, 2})
    D, M, N = dmn(X, Y)
    σ² = 0.0
    for n = 1:N
        for m = 1:M
            @views σ² += sum(x->x^2, X[:, n] - Y[:, m])
        end
    end
    return σ² / (D * M  *N)
end

# eqn. (7) in paper
function q_objective(X::Array{Float64, 2}, Y::Array{Float64, 2}, P::Array{Float64, 2},
                     σ²::Float64, R::Array{Float64, 2}, t::Array{Float64, 1})
    D, M, N = dmn(X, Y)
    Np = sum(P)

    Y_transformed = R * Y .+ t

    q = 0.0
    for n = 1:N
        for m = 1:M
            @inbounds @views q += P[m, n] * sum(x->x^2, X[:, n] - Y_transformed[:, m])
        end
    end
    q /= 2 * σ²
    q += Np * D / 2 * log(σ²)
    return q
end

"""
    rigid_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2};
                                 allow_translation, max_nb_em_steps,
                                 q_tol, w, verbose, print_ending, σ²_tol)

What transformation R * Y + t needs to be applied to Y to make it match X?

# Returns
    rotation matrix, translation vector, variance, error
"""
function rigid_point_set_registration(X::Array{Float64, 2}, Y::Array{Float64, 2};
                                      allow_translation::Bool=false,
                                      max_nb_em_steps::Int=25, q_tol::Float64=1e-5,
                                      w::Float64=0.0, verbose::Bool=true, print_ending::Bool=true,
                                      σ²_tol::Float64=1e-6)
    D, M, N = dmn(X, Y)

    start_time = time()
    
    # all of these will be updated in the loop.
    # the rotation matrix
    R = diagm(0 => [1.0 for i = 1:D])
    # the translation vector
    t = zeros(D)
    # the bandwidth (variance) of the Gaussian mixture model
    σ² = initial_σ²(X, Y)
    # probabilities of correspondence
    P = zeros(Float64, M, N)
    # the objective function, new and old
    q = 0.0
    q_old = Inf
    # the transformed Y data
    Y_transformed = R * Y .+ t
    # why did we break the loop?
    reason_for_exit = "max EM steps reached"

    em_step = 0

    while em_step < max_nb_em_steps
        em_step += 1
        # conduct the rotation with our best guess of R
        Y_transformed .= R * Y .+ t

        # E step. Get correspondence.
        for n = 1:N
            for m = 1:M
                @inbounds @views P[m, n] = exp(-sum(x->x^2, Y_transformed[:, m] - X[:, n]) / (2 * σ²))
            end
            # at this pt row m is filled
            @inbounds @views P[:, n] ./= sum(P[:, n]) + (2*π*σ²)^(D/2) * w / (1.0 - w) * M / N
        end
        Np = sum(P)
        
        # M step. Get best transformation
        μx = (sum(X * P', dims=2) / Np)[:]
        μy = (sum(Y * P, dims=2) / Np)[:]
 #         μx = X * P' * ones(M) / Np
 #         μy = Y * P * ones(N) / Np
        Xhat = broadcast(-, X, μx)
        Yhat = broadcast(-, Y, μy)

        A = Xhat * P' * Yhat'

        F = svd(A)
        
        # update transformation
        R = F.U * diagm(0 => [i == D ? det(F.U * F.Vt) : 1.0 for i = 1:D]) * F.Vt

        if allow_translation
            t .= μx .- R * μy
        end
        
        # update variance
        σ² = (tr(Xhat * diagm(0 => sum(P', dims=2)[:]) * Xhat') - tr(A' * R)) / (Np * D)
        if σ² < 0.0
            σ² = σ²_tol * 2.0
        end

        # objective
        q = q_objective(X, Y, P, σ², R, t)
        if verbose
            println("\tEM step: ", em_step)
            println("\t\tobjective: ", q)
            println("\t\tσ² = ", σ²)
        end
        
        # terminate if objective hasn't decreased much, suggesting convergence
        if abs(q - q_old) < q_tol
            reason_for_exit = "objective stopped decreasing"
            break
        end
        if σ² < σ²_tol
            reason_for_exit = "variance below tol"
            break
        end

        q_old = q
    end
    
    if print_ending
        @printf("\tσ² = %f, q = %f, EM steps taken = %d, reason for exit: %s, time: %.5f min\n",
            σ², q, em_step, reason_for_exit, (time() - start_time) / 60)
    end

    return R, t, σ², q
end

export rigid_point_set_registration

end
