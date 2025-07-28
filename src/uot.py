import numpy as np

def unbalanced_ot_solver(C, P_minus, P_plus, lambda_1, lambda_2, 
                         eta_alpha, eta_beta, epsilon, T, 
                         init_alpha=None, init_beta=None):
    """
    Gradient-based Unbalanced Optimal Transport solver.

    Parameters
    ----------
    C : ndarray of shape (m, n)
        Cost matrix C[i,j] = c(x_i, y_j).
    P_minus : ndarray of shape (m,)
        Distribution P_-(x_i).
    P_plus : ndarray of shape (n,)
        Distribution P_+(y_j).
    lambda_1 : float
        Regularization parameter for KL divergence on row marginals.
    lambda_2 : float
        Regularization parameter for chi^2 divergence on column marginals.
    eta_alpha : float
        Step size for gradient descent in alpha.
    eta_beta : float
        Step size for gradient descent in beta.
    epsilon : float
        Small regularization parameter for the exponential kernel.
    T : int
        Number of iterations.
    init_alpha : ndarray of shape (m,), optional
        Initial alpha potentials. Default is zeros.
    init_beta : ndarray of shape (n,), optional
        Initial beta potentials. Default is zeros.

    Returns
    -------
    pi : ndarray of shape (m, n)
        Final transport plan.
    alpha : ndarray of shape (m,)
        Final dual potential alpha.
    beta : ndarray of shape (n,)
        Final dual potential beta.
    """

    m, n = C.shape
    
    # Initialize alpha and beta
    if init_alpha is None:
        alpha = np.zeros(m)
    else:
        alpha = init_alpha.copy()
    if init_beta is None:
        beta = np.zeros(n)
    else:
        beta = init_beta.copy()

    for t_iter in range(T):
        # Update transport plan pi
        # pi(i,j) ~ exp(-(c_ij - alpha_i - beta_j)/epsilon)
        exponent = -(C - alpha[:, None] - beta[None, :]) / epsilon
        max_exponent = np.max(exponent)  # for numerical stability
        exp_exponent = np.exp(exponent - max_exponent)
        Z = np.sum(exp_exponent)
        pi = exp_exponent / Z

        # Compute marginals
        pi_1 = np.sum(pi, axis=1)  # row sum
        pi_2 = np.sum(pi, axis=0)  # column sum

        # Compute objective derivatives
        # Objective:
        # F = sum_{i,j} pi(i,j)*C[i,j] 
        #     + lambda_1 * D_KL(pi_1 || P_minus)
        #     + lambda_2 * D_chi2(pi_2 || P_plus)
        #
        # D_KL(r||s) = sum_i r_i log(r_i/s_i)
        # D_chi2(r||s) = sum_j ((r_j - s_j)^2 / s_j)

        # We'll need gradients w.r.t alpha and beta.
        # pi depends on alpha,beta as:
        # pi(i,j) = exp(-(c_ij - alpha_i - beta_j)/epsilon) / Z
        #
        # Analytically, we have:
        # d pi(i,j)/d alpha(i) and d pi(i,j)/d beta(j).
        # Insert your derived formulas here.

        # Placeholder gradients for demonstration:
        # The gradient computations below are schematic.
        # In a proper implementation, you'd replace these with
        # the correct analytical partial derivatives.
        
        # Compute partial derivatives w.r.t. pi first:
        # dF/dpi(i,j) = c_ij 
        #    + lambda_1 * d/dpi(i,j)[D_KL(pi_1||P_-)]
        #    + lambda_2 * d/dpi(i,j)[D_chi2(pi_2||P_+)]

        # For KL:
        # D_KL(pi_1||P_-) = sum_i [pi_1(i)*log(pi_1(i)/P_minus(i)) - pi_1(i) + P_minus(i)]
        # dD_KL/dpi_1(i) = log(pi_1(i)/P_minus(i)) + 1 - 1 = log(pi_1(i)/P_minus(i))
        # => dF/dpi(i,j) includes lambda_1 * log(pi_1(i)/P_minus(i))
        
        # For chi^2:
        # D_chi2(pi_2||P_+) = sum_j ((pi_2(j)-P_plus(j))^2 / P_plus(j))
        # dD_chi2/dpi_2(j) = 2(pi_2(j)-P_plus(j))/P_plus(j)
        # => dF/dpi(i,j) includes lambda_2 * 2(pi_2(j)-P_plus(j))/P_plus(j)

        dF_dpi = C \
                 + lambda_1 * (np.log(pi_1 / P_minus)[:, None]) \
                 + lambda_2 * (2 * (pi_2 - P_plus)[None, :] / P_plus[None, :])

        # Now we convert dF/dpi to dF/dalpha(i) and dF/dbeta(j).
        # Recall pi(i,j) = exp(...) / Z, 
        # d pi(i,j)/d alpha(i) = (See note below)
        
        # Note: 
        # d pi(i,j)/d alpha(i) = pi(i,j)*(-1/epsilon - sum_{k,l} pi(k,l)*(-1/epsilon))
        # But sum_{k,l} pi(k,l) = 1, so:
        # d pi(i,j)/d alpha(i) = pi(i,j)*(-1/epsilon + 1/epsilon*1) = 0 if computed this way?
        # This seems suspicious. Let's carefully consider the dependence:
        #
        # Actually, since alpha appears with a minus sign in the exponent,
        # d/d alpha(i) [-(c_{ij}-alpha_i-beta_j)/epsilon] = 1/epsilon.
        #
        # More directly:
        # Let's define Q(i,j) = -(c_ij - alpha_i - beta_j)/epsilon
        # pi(i,j) = exp(Q(i,j))/Z
        # d pi(i,j)/d alpha(i) 
        # = (d/d alpha(i) exp(Q(i,j)))*1/Z - exp(Q(i,j))*(1/Z^2)*dZ/d alpha(i)
        # = exp(Q(i,j))/Z * (1/epsilon) - pi(i,j)*(1/Z)*dZ/d alpha(i)
        #
        # dZ/d alpha(i) = sum_l (d/d alpha(i) exp(Q(i,l)))
        # = sum_l exp(Q(i,l))*(1/epsilon) = (1/epsilon)*Z*pi_1(i)
        #
        # Thus:
        # d pi(i,j)/d alpha(i) 
        # = pi(i,j)*(1/epsilon) - pi(i,j)*pi_1(i)*(1/epsilon)
        # = (pi(i,j)/epsilon)*(1 - pi_1(i))
        #
        # Similarly:
        # d pi(i,j)/d beta(j)
        # = (pi(i,j)/epsilon)*(1 - pi_2(j))

        # Given this, we can find dF/d alpha(i):
        # dF/d alpha(i) = sum_j dF/dpi(i,j)*d pi(i,j)/d alpha(i)
        # = sum_j dF/dpi(i,j) * (pi(i,j)/epsilon)*(1 - pi_1(i))
        # Factor (1 - pi_1(i))/epsilon out:
        # dF/d alpha(i) = (1 - pi_1(i))/epsilon * sum_j (pi(i,j)*dF/dpi(i,j))

        # Similarly for beta(j):
        # dF/d beta(j) = (1 - pi_2(j))/epsilon * sum_i (pi(i,j)*dF/dpi(i,j))

        # Compute dF/d alpha:
        dF_dalpha = np.zeros(m)
        for i in range(m):
            dF_dalpha[i] = (1 - pi_1[i])/epsilon * np.sum(pi[i,:]*dF_dpi[i,:])

        # Compute dF/d beta:
        dF_dbeta = np.zeros(n)
        for j in range(n):
            dF_dbeta[j] = (1 - pi_2[j])/epsilon * np.sum(pi[:,j]*dF_dpi[:,j])

        # Update alpha and beta
        alpha -= eta_alpha * dF_dalpha
        beta -= eta_beta * dF_dbeta

    return pi, alpha, beta

def unbalanced_ot_kl_chi2(a, b, C, lambda_1=1.0, lambda_2=1.0, epsilon=1e-2, max_iter=1000, tol=1e-9):
    """
    Solve the unbalanced OT problem:
    min_{P >= 0} <C,P> + lambda_1 * KL(P1||a) + lambda_2 * chi2(P^T1 || b)
    using a generalized Sinkhorn-like algorithm with entropic smoothing.

    Parameters
    ----------
    a : ndarray, shape (n,)
        Source distribution (must be positive)
    b : ndarray, shape (m,)
        Target distribution (must be positive)
    C : ndarray, shape (n, m)
        Cost matrix
    lambda_1 : float
        Weight for KL divergence on the source side
    lambda_2 : float
        Weight for chi2 divergence on the target side
    epsilon : float
        Regularization parameter for entropic smoothing
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns
    -------
    P : ndarray, shape (n, m)
        Optimal transport plan
    """

    n, m = C.shape
    # Exponential kernel
    K = np.exp(-C / epsilon)

    # Initialize scaling factors
    u = np.ones(n) / n
    v = np.ones(m) / m

    for it in range(max_iter):
        # Compute current plan
        P = (u[:, None] * K) * v[None, :]
        # Marginals of current plan
        r = P.sum(axis=1)  # should approach a (in the KL sense)
        c = P.sum(axis=0)  # should approach b (in the chi2 sense)

        # Store old variables for convergence check
        u_old = u.copy()
        v_old = v.copy()

        # Update u for the KL divergence
        # From the dual condition, a stable fixed-point step:
        # We want r_i close to a_i in KL sense. For small steps:
        # KL condition: log(r_i/a_i) ~ correction
        # A heuristic update:
        #u = u * np.exp(- (lambda_1/epsilon) * (np.log(np.maximum(r / a, 0.001))))

        # Update v for the chi2 divergence
        # chi2 derivative w.r.t. c: d/dc_j = 2(c_j - b_j)/b_j
        # We'll take a small step in the dual variable:
        # For stability, consider half-step or a small factor. Here we use a direct exponentiation.
        #v = v * np.exp(- (lambda_2/(epsilon)) * ((c - b)/b))
        alpha = 0.1  # for instance, half step
        log_ratio = np.log(r / a)
        log_ratio = np.clip(log_ratio, -10, 10)  # Avoid too large magnitudes
        u = u * np.exp(- alpha * (lambda_1/epsilon) *  log_ratio)
        chi_ratio = (c - b)/b
        chi_ratio = np.clip(chi_ratio, -10, 10)
        v = v * np.exp(- alpha * (lambda_2/epsilon) * chi_ratio)
        
        # Check convergence
        diff_u = np.max(np.abs(u - u_old))
        diff_v = np.max(np.abs(v - v_old))
        if max(diff_u, diff_v) < tol:
            break

    P = (u[:, None] * K) * v[None, :]
    return P
