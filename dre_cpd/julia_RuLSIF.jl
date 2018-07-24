using NPZ, PyPlot

function comp_med(x)
    d, n = size(x)
    G = sum(x.*x, 1)
    T = repmat(G, n, 1)
    dist2 = T - 2*x'*x + T'
    dist2 = dist2 - tril(dist2)
    R = dist2[:]
    med = sqrt(0.5 * median(R[R.>0]))
    return med
end

function comp_dist(x, y)
    d, nx = size(x)
    d, ny = size(y)

    G = sum(x.*x, 1)
    T = repmat(G, ny, 1)
    G = sum(y.*y, 1)
    R = repmat(G, nx, 1)

    dist2 = T' + R - 2.*x'*y
    return dist2
end

function kernel_rbf(dist2, ℓ)
    k = exp.(-dist2/(2*ℓ^2))
    return k
end

function sliding_window(X, window_size, step)
    n_dims, n_samples = size(X)
    WINDOWS = zeros(n_dims * window_size * step, floor(n_samples/window_size/step))
    
    n_cols = size(WINDOWS, 2) 
    for i = 1:step:n_samples
        offset = window_size * step
        if i + offset - 1 > n_samples
            break
        end
        w = X[:, i:i+offset-1']'
        
        if Int(ceil(i/step)) > n_cols
            WINDOWS = cat(2, WINDOWS, w[:])
        else
            WINDOWS[:, Int(ceil(i/step))] = w[:]
        end
    end
    return WINDOWS
end

function RuLSIF(x_de, x_nu, α, fold)
    _, n_nu = size(x_nu)
    _, n_de = size(x_de)

    b = min.(100, n_nu)
    idx = randperm(n_nu)
    x_ce = x_nu[:, idx[1:b]]

    _, n_ce = size(x_ce)
    x = cat(2, x_de, x_nu)
    med = comp_med(x)
    ℓ_list = med * [0.6, 0.8, 1.0, 1.2, 1.4]
    λ_list = 10.0.^collect(-3:1:1)

    dist2_de = comp_dist(x_de, x_ce)
    dist2_nu = comp_dist(x_nu, x_ce)
    
    score = zeros(length(ℓ_list), length(λ_list))
    for i = 1:length(ℓ_list)
        k_de = kernel_rbf(dist2_de, ℓ_list[i])
        k_nu = kernel_rbf(dist2_nu, ℓ_list[i])
        
        for j = 1:length(λ_list)
            cv_index_nu = randperm(n_nu)
            cv_split_nu = floor.(collect(0:1:n_nu-1)*fold./n_nu) + 1
            cv_index_de = randperm(n_de)
            cv_split_de = floor.(collect(0:1:n_de-1)*fold./n_de) + 1
            
            sum = 0
            for k = 1:fold
                k_de_k = k_de[cv_index_de[cv_split_de .!= k], :]'
                k_nu_k = k_nu[cv_index_nu[cv_split_nu .!= k], :]'
                H_k = ((1-α)/size(k_de_k,2)) * k_de_k * k_de_k' + (α/size(k_nu_k,2)) * k_nu_k * k_nu_k'
                h_k = mean(k_nu_k, 2)
                θ = (H_k + eye(n_ce) * λ_list[j])\h_k
                k_de_test = k_de[cv_index_de[cv_split_de .== k], :]'
                k_nu_test = k_nu[cv_index_nu[cv_split_nu .== k], :]'
                
                # objective function value
                J = α/2 * mean((θ' * k_nu_test).^2) + (1-α)/2 * mean((θ' * k_de_test).^2) - mean(θ' * k_nu_test)
                sum = sum + J
            end
            score[i, j] = sum/fold
        end
    end
    i_min, j_min = ind2sub(size(score), indmin(score))
    ℓ_optimal = ℓ_list[i_min]
    λ_optimal = λ_list[j_min]

    k_de = kernel_rbf(dist2_de', ℓ_optimal)
    k_nu = kernel_rbf(dist2_nu', ℓ_optimal)

    H = ((1-α)/n_de) * k_de * k_de' + (α/n_nu) * k_nu * k_nu'
    h = mean(k_nu, 2)

    θ = (H + eye(n_ce) * λ_optimal)\h

    g_nu = θ' * k_nu
    g_de = θ' * k_de
    g_re = []                   # further, this will be implemeneted

    rPE = mean(g_nu) - 0.5*(α*mean(g_nu.^2) + (1-α)*mean(g_de.^2)) - 0.5
    return rPE, g_nu, g_re, ℓ_optimal, λ_optimal
end

function changepoint_detection(X, n, k, α, fold)
    SCORE = []

    WIN = sliding_window(X, k, 1)
    n_samples = size(WIN, 2)
    t = n + 1
    ℓ_track = []
    λ_track = []

    while t + n - 1 ≤ n_samples
        Y = WIN[:, t-n : n+t-1]
        Y = Y./repmat(std(Y,2), 1, 2*n)
        Yref = Y[:, 1:n]
        Ytest = Y[:, n+1:end]

        s, _, _, ℓ, λ = RuLSIF(Yref, Ytest, α, fold)
        push!(ℓ_track, ℓ)
        push!(λ_track, λ)

        if (mod(t, 20) == 0)
            println(t, "th loop finished")
        end

        push!(SCORE, s)
        t += 1
    end
end

function demo()
    srand(1)
    n = 50
    k = 10
    α = 0.01
    y = npzread("logwell.npy")

    score1 = changepoint_detection(y, n, k, α, 5)
    score2 = changepoint_detection(y[:, end:-1:1], n, k, α, 5)
    
    score = score1 + score2
    plot(collect(1:1:length(score)), score, color="red", linewidth=2.0, linestyle='-')
    title("Change-Point Score")
end

demo()
