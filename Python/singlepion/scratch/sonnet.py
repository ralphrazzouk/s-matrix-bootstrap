import numpy as np
import cvxpy as cp

# Define the number of interpolation points
M = 100

# Define the grid of interpolation points
xi_j = np.linspace(0, np.pi, M + 1)[:-1]    # Exclude the last point (pi)
s_j = 4 / np.cos(xi_j/2)**2                 # Map to the s-plane

# Define the kernels
def K(xi, s, epsilon=1e-12):
    return (1/np.pi) * (np.sin(xi) / (8/s - 1 - np.cos(xi) + epsilon))

# Define the functions needed for the partial waves
def Phi_l(s, xi_j):
    mu = np.linspace(-1, 1, 1001)  # Dense grid for numerical integration
    t = -(s - 4) * (1 - mu) / 2
    u = 4 - s - t
    integrand = np.polynomial.legendre.legendre(l)(mu) * K(xi_j, t)
    return np.trapz(integrand, mu)

def Phi_tilde_l(s, xi_j1, xi_j2):
    mu = np.linspace(-1, 1, 1001)
    t = -(s - 4) * (1 - mu) / 2
    u = 4 - s - t
    integrand = np.polynomial.legendre.legendre(l)(mu) * K(xi_j1, t) * K(xi_j2, u)
    return np.trapz(integrand, mu)

# Define the coefficients a
a_0 = 1
a_j1 = np.pi * (K(xi_j1, s_0) + K(xi_j1, t_0) + K(xi_j1, u_0))
a_j1j2 = (np.pi**2 / 2) * (K(xi_j1, s_0) * K(xi_j2, t_0) + K(xi_j1, s_0) * K(xi_j2, u_0) +
                           K(xi_j1, t_0) * K(xi_j2, u_0) + K(xi_j2, s_0) * K(xi_j1, t_0) +
                           K(xi_j2, s_0) * K(xi_j1, u_0) + K(xi_j2, t_0) * K(xi_j1, u_0))

# Define the partial waves h
def h_lj(l):
    return (np.pi / 2) * np.sin(xi_j / 2) * np.eye(M+1)[l, 0]

def h_lj_j1(l, j1):
    return (np.pi / 2) * np.sin(xi_j / 2) * (np.eye(M+1)[l, 0] * K(xi_j1, s_j) + Phi_l(s_j, xi_j1))

def h_lj_j1j2(l, j1, j2):
    return (np.pi / 2) * np.sin(xi_j / 2) * (K(xi_j1, s_j) * Phi_l(s_j, xi_j2) +
                                            (1/2) * Phi_tilde_l(s_j, xi_j1, xi_j2) +
                                            (1/2) * Phi_tilde_l(s_j, xi_j2, xi_j1))

# Define the primal problem
f_0 = cp.Variable(1)
sigma_j1 = cp.Variable(M)
rho_j1j2 = cp.Variable((M, M))

constraints = []
for j in range(M+1):
    for l in range(lmax+1):
        if l % 2 == 0:  # Only even l contribute
            h_l = h_lj(l) * f_0 + cp.sum(h_lj_j1(l, j1) * sigma_j1[j1] for j1 in range(M)) + \
                  cp.sum(h_lj_j1j2(l, j1, j2) * rho_j1j2[j1, j2] for j1 in range(M) for j2 in range(M))
            constraints += [cp.abs(1 + 1j * h_l) <= 1]

obj = a_0 * f_0 + cp.sum(a_j1 * sigma_j1) + cp.sum(a_j1j2 * rho_j1j2)
prob = cp.Problem(cp.Maximize(obj), constraints)
prob.solve()

# Access the solution
print(f"Maximum value of the amplitude: {prob.value}")
print(f"Optimal f_0: {f_0.value}")
print(f"Optimal sigma: {sigma_j1.value}")
print(f"Optimal rho: {rho_j1j2.value}")