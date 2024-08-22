import numpy as np
import cvxpy as cp



def compute_kernel(xi, s):
    """
    Compute the kernel K(xi, s).
    """
    return (1/np.pi) * (np.sin(xi) / (8/s - 1 - np.cos(xi)))

def compute_phi_integrals(s, xi):
    """
    Compute the integrals Phi_l(s; xi) and Phi_l(s; xi_1, xi_2).
    """
    mu = np.linspace(-1, 1, 100)
    # t = - (s - 4) * (1 - mu)/2
    # u = 4 - s + (s - 4) * (1 - mu)/2

    def phi_l(s, xi):
        P_l = np.polynomial.legendre.Legendre.basis(len(mu) - 1)(mu)
        integrand = P_l * compute_kernel(xi, - (s - 4) * (1 - mu)/2)
        return np.trapz(integrand, mu, axis = -1)

    def phi_tilde_l(s, xi_1, xi_2):
        P_l = np.polynomial.legendre.Legendre.basis(len(mu) - 1)(mu)
        integrand = P_l * compute_kernel(xi_1, 4 - s + (s - 4) * (1 - mu)/2) * compute_kernel(xi_2, 4 - s + (s - 4) * (1 - mu)/2)
        return np.trapz(integrand, mu, axis = -1)

    phi_l_vals = np.array([phi_l(s, xi_j) for xi_j in xi])
    phi_tilde_l_vals = np.array([[phi_tilde_l(s, xi_j1, xi_j2) for xi_j2 in xi] for xi_j1 in xi])

    return phi_l_vals, phi_tilde_l_vals


def interpolate_values(M):
    """
    Compute the interpolation points and values.
    """
    j = 1/2 + np.arange(0, M - 1)
    delta_xi = np.pi / M
    xi = j * delta_xi
    s = 4 / np.cos(xi/2)**2

    return delta_xi, xi, s


M = 30
delta_xi, xi, s = interpolate_values(M)


# Compute the coefficients
def coefficients(xi, s, t, u):
    a = 1

    def a_j1(xi_j1, s, t, u):
        return delta_xi * (compute_kernel(xi_j1, s) + compute_kernel(xi_j1, t) + compute_kernel(xi_j1, u))

    def a_j1j2(xi_j1, xi_j2, s, t, u):
        term1 = 0.5 * delta_xi**2 * (compute_kernel(xi_j1, s) * compute_kernel(xi_j2, t) +
                                        compute_kernel(xi_j1, s) * compute_kernel(xi_j2, u) +
                                        compute_kernel(xi_j1, t) * compute_kernel(xi_j2, u))

        term2 = 0.5 * delta_xi**2 * (compute_kernel(xi_j2, s) * compute_kernel(xi_j1, t) +
                                        compute_kernel(xi_j2, s) * compute_kernel(xi_j1, u) +
                                        compute_kernel(xi_j2, t) * compute_kernel(xi_j1, u))

        return term1 + term2

    a_vals = np.array([a for xi_j in xi])
    a_j1_vals = np.array([a_j1(xi_j1, s, t, u) for xi_j1 in xi])
    a_j1j2_vals = np.array([[a_j1j2(xi_j1, xi_j2, s, t, u) for xi_j2 in xi] for xi_j1 in xi])

    return a_vals, a_j1_vals, a_j1j2_vals

a_vals, a_j1_vals, a_j1j2_vals = coefficients(xi, s0, t0, u0)




# Compute the partial waves
phi_l_vals, phi_tilde_l_vals = compute_phi_integrals(s, xi)
def partial_waves(l, xi, s_jplus):
    def h_lj(l, xi_j):
        return (np.pi/2) * np.sin(xi_j/2) * (l == 0)

    def h_lj_j1(l, xi_j, xi_j1, s_jplus):
        return (np.pi/2) * np.sin(xi_j/2) * ((l == 0) * compute_kernel(xi_j1, s_jplus) + phi_l_vals)

    def h_lj_j1j2(l, xi_j, xi_j1, xi_j2, s_jplus):
        term1 = (np.pi/2) * np.sin(xi_j/2) * (compute_kernel(xi_j1, s_jplus) * delta_xi * phi_l_vals + 0.5 * delta_xi**2 * phi_tilde_l_vals)
        term2 = (np.pi/2) * np.sin(xi_j/2) * (compute_kernel(xi_j2, s_jplus) * delta_xi * phi_l_vals + 0.5 * delta_xi**2 * phi_tilde_l_vals)

        return term1 + term2

    h_lj_vals = np.array([h_lj(l, xi_j) for xi_j in xi])
    h_lj_j1_vals = np.array([[h_lj_j1(l, xi_j, xi_j1, s_jplus) for xi_j1 in xi] for xi_j in xi])
    h_lj_j1j2_vals = np.array([[[h_lj_j1j2(l, xi_j, xi_j1, xi_j2, s_jplus) for xi_j2 in xi] for xi_j1 in xi] for xi_j in xi])

    return h_lj_vals, h_lj_j1_vals, h_lj_j1j2_vals

    # h_lj = np.pi/2 * np.sin(xi/2) * (l == 0)
    # h_lj_j1 = np.pi/2 * np.sin(xi/2) * phi_l_vals
    # h_lj_j1j2 = np.pi/2 * np.sin(xi/2)[:, np.newaxis, np.newaxis] * phi_tilde_l_vals
    # h_lj_j1j2 += h_lj_rho.transpose(0, 2, 1)

h_lj_vals, h_lj_j1_vals, h_lj_j1j2_vals = partial_waves(lmax, xi, s0)
h_l = np.concatenate((h_lj_vals.flatten(), h_lj_j1_vals.flatten(), h_lj_j1j2_vals.flatten()))






def primal_problem(M, s0, t0, u0, lmax, M_reg):
    """
    Solve the primal problem.
    """

    # Define the primal variables
    f0 = cp.Variable()
    sigma = cp.Variable(M + 1)
    rho = cp.Variable((M + 1, M + 1), symmetric=True)




    # h_l = []
    # for l in range(0, lmax + 1, 2):
    #     phi_l_vals, phi_tilde_l_vals = compute_phi_integrals(s, xi)
    #     h_lj_f0 = np.pi/2 * np.sin(xi/2) * (l == 0)
    #     h_lj_sigma = np.pi/2 * np.sin(xi/2) * phi_l_vals
    #     h_lj_rho = np.pi/2 * np.sin(xi/2)[:, np.newaxis, np.newaxis] * phi_tilde_l_vals
    #     h_lj_rho += h_lj_rho.transpose(0, 2, 1)  # Add the (j1 <-> j2) term

    #     h_l.append(cp.sum(h_lj_f0) * f0 + cp.sum(h_lj_sigma * sigma) +

    # Define the optimization problem
    objective = cp.Maximize(a_vals*f0 + cp.sum(cp.multiply(a_j1_vals, sigma)) + cp.sum(cp.multiply(a_j1j2_vals, rho)))
    constraints = [cp.abs(1 + 1j*h_l[l//2]) <= 1 for l in range(0, lmax + 1, 2)]
    constraints += [cp.norm(cp.hstack([f0, sigma, cp.reshape(rho, ((M + 1)**2, 1))])) <= M_reg]
    prob = cp.Problem(objective, constraints)

    # Solve the optimization problem
    prob.solve()

    # Retrieve the optimal value and variables
    F_P_max = prob.value
    f0_opt = f0.value
    sigma_opt = sigma.value
    rho_opt = rho.value

    return F_P_max, f0_opt, sigma_opt, rho_opt

# Example usage
M_values = [9]  # Number of interpolation points
s0, t0, u0 = 4/3, 4/3, 4/3  # Symmetric point
lmax = 10  # Maximum angular momentum
M_reg_values = [1e2, 1e3, 1e4, 1e5]  # Regularization parameter values

for M in M_values:
    for M_reg in M_reg_values:
        F_P_max, f0_opt, sigma_opt, rho_opt = primal_problem(M, s0, t0, u0, lmax, M_reg)
        print(f"Primal maximum (M={M}, M_reg={M_reg}): {F_P_max}")