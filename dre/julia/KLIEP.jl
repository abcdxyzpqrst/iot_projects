##### KL-divergence Density-Ratio Estimation #####
using Roots, PyPlot

"""
Change-Point Detection in Time-Series Data by Direct Density-Ratio Estimation

Done:   Algorithm 1,2

KLIEP algorithm --> Algorithm 1
Kernel Lengthscale Selection by CV --> Algorithm 2 (TODO: by hyperparameter learning?)

Online Version --> Algorithm 3,4
"""

function Gaussian_pdf(x, μ, Σ)
    D, N = size(x)
    prob = 0.0
    return prob
end

function RBF(X1, X2; ℓ=1.0)
    """
    RBF Kernel
    X1, X2 ==> design matrices
    ℓ:  lengthscale
    """
    D, N = size(X1)
    D, M = size(X2)
    
    X1s = sum(X1.^2, 1)
    X2s = sum(X2.^2, 1)

    sdist = repmat(X2s, N, 1) + repmat(X1s', 1, M) - 2*X1'*X2
    K = exp.(-sdist / (2*ℓ^2))
    return K
end

function KLIEP_projection(α, Xte, b, c)
    """
    Perform feasibility satisfaction
    """
    α = α + b*(1-sum(b.*α))*pinv(c)     # α <- α + (1 - bᵀα)b / c, c := bᵀb
    α = max.(0, α)                      # α <- max(0, α)
    α = α * pinv(sum(b.*α))             # α <- α / bᵀα
    Xte_α = Xte * α
    score = mean(log.(Xte_α))
    return α, Xte_α, score
end

function KLIEP_solve(mean_X_de, X_nu)
    n_nu, nc = size(X_nu)

    itm = 100
    η_list = 10.0.^(3:-1:-3)
    c = sum(mean_X_de.^2)
    α = ones(nc, 1)
    α, X_nu_α, score = KLIEP_projection(α, X_nu, mean_X_de, c)
    
    for η in η_list
        for ite = 1:itm
            ∇α = X_nu'*(1./X_nu_α)
            α_t = α + η*∇α
            α_new, X_nu_α_new, score_new = KLIEP_projection(α_t, X_nu, mean_X_de, c)
            
            if score_new - score ≤ 0
                break
            end

            score = score_new
            α = α_new
            X_nu_α = X_nu_α_new
        end
    end
    return α, score
end

function KLIEP(x_nu, x_de; x_re=0, ℓ_optimal=0.0, b=100, fold=5)
    """
    KL-divergence Importance Estimation Procedure with Cross-Validation

    Estimating Probability Density-Ratio
        p_nu(x)/p_de(x)
    
    From samples
        
    """
    d, n_de = size(x_de)
    d_nu, n_nu = size(x_nu)
    
    d != d_nu && error("x_nu & x_de must have same dimension!!")

    randix = randperm(n_nu)
    b = min(b, n_nu)
    x_ce = x_nu[:, randix[1:b]]         # centers for Gaussian dist

    if ℓ_optimal == 0.0
        ℓ = 10.0
        score = -Inf

        for ϵ = log10(ℓ)-1:-1:-5
            for iter = 1:9
                ℓ_new = ℓ - 10^ϵ

                cv_index = randperm(n_nu)
                cv_split = floor.(collect(0:1:n_nu-1)*fold./n_nu) + 1
                score_new = 0

                X_nu = RBF(x_nu, x_ce; ℓ=ℓ_new)
                X_de = RBF(x_de, x_ce; ℓ=ℓ_new)
                mean_X_de = mean(X_de, 1)'
                
                for i = 1:fold
                    α_cv = KLIEP_solve(mean_X_de, X_nu[cv_index[cv_split .!= i], :])
                    println(α_cv)
                    wh_cv = X_nu[cv_index[cv_split .!= i], :] * α_cv
                    score_new = score_new + mean(log(wh_cv))/fold
                end

                if (score_new - score) ≤ 0
                    break
                end
                score = score_new
                ℓ = ℓ_new
            end
        end
        ℓ_optimal = ℓ
    end
    
    X_nu = RBF(x_nu, x_ce; ℓ=ℓ_optimal)
    X_de = RBF(x_de, x_ce; ℓ=ℓ_optimal)
    mean_X_de = mean(X_de, 1)'
    αh = KLIEP_learning(mean_X_de, X_nu)
    wh_x_de = (X_de * αh)
    
    if x_re == 0
        wh_x_re = NaN
    else
        X_re = RBF(x_re, x_ce; ℓ=ℓ_optimal)
        wh_x_re = (X_re * αh)
    end
    return wh_x_de, wh_x_re
end

function main(args="")
    # create data for demo
    srand(12345678)
    
    # dimension
    d = 1

    # samples from numerator probability density
    n_nu = 100
    μ_nu = 1
    σ_nu = 1/8

    # samples from denominator probability density
    n_de = 100
    μ_de = 1
    σ_de = 1/2

    x_nu = μ_nu + σ_nu*randn(d, n_nu)
    x_de = μ_de + σ_de*randn(d, n_de)

    KLIEP(x_nu, x_de)
    
    # plotting results
end

PROGRAM_FILE == "KLIEP.jl" && main(ARGS)
