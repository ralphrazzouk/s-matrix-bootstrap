import numpy as np
import scipy as sp
import sympy as smp
import mpmath as mp
# import cvxpy as cp
# from decimal import Decimal, getcontext
# getcontext().prec = 10000
# Decimal(10)**-100
# Decimal('1E-100')

print("\n")


M = 3                                                  # Number of interpolation points
xi = np.linspace(0, 2*np.pi, 2*M, endpoint = False)     # Interpolation points
# print("xi", xi)
s = 4 / np.cos(xi/2)**2                                 # Mapping from xi to s

# Compute the kernel K(xi, s)
K = np.zeros((2*M, 2*M))
for j1 in range(0, 2*M):
    for j2 in range(j1, 2*M):
        K[j1, j2] = 1/(2*M) * (1 - (-1)**(j1 - j2)) * 1/np.tan(np.pi/(2*M) * (j1 - j2))
        K[j2, j1] = - K[j1, j2]

        np.nan_to_num(K, copy=False)

# print("K_j1j2:", K)


K_i = np.zeros((M + 1, M + 1))
K_d = np.zeros((M + 1, M + 1))
for j1 in range(0, M + 1):
    # print("j1:", j1)
    for j2 in range(1, M):
        # print("j2:", j2)
        # print("formula:", 2*M - j2)
        K_i[j1, j2] = - (K[j1, j2] - K[j1, 2*M - j2])
        K_d[j1, j2] = K[j1, j2] + K[j1, 2*M - j2]

for j1 in range(0, M + 1):
    K_d[j1, 0] = K[j1, 0]
    K_d[j1, M] = K[j1, M]

# print("K_i:", K_i)
# print("K_d:", K_d)





fvra = np.cos(4*xi)
fvia = np.sin(4*xi)
fviaN = np.dot(K, fvra)
norm1 = np.linalg.norm(fvia - fviaN)
# print("fvra:", fvra)
# print("fvia:", fvia)
# print("fviaN:", fviaN)
print("norm1:", norm1)

# print("\n")

fvr = np.cos(4*xi)[:M+1]
fvi = np.sin(4*xi)[:M+1]
fvrN = np.dot(K_i, fvi)
fviN = np.dot(K_d, fvr)
norm2 = np.linalg.norm(fvr - fvrN)
norm3 = np.linalg.norm(fvi - fviN)
# print("fvr:", fvr)
# print("fvi:", fvi)
# print("fvrN:", fvrN)
# print("fviN:", fviN)
print("norm2:", norm2)
print("norm3:", norm3)





lmax = 20
s0, t0, u0 = 4/3, 4/3, 4/3

def mathcalK(xi, s):
    return 1/np.pi * np.sin(xi) / (8/s - 1 - np.cos(xi))

as1 = mathcalK(xi, s0)
at1 = mathcalK(xi, t0)
au1 = mathcalK(xi, u0)

a0 = 1
a1 = as1 + at1 + au1
a2 = np.outer(as1, at1) + np.outer(as1, au1) + np.outer(at1, au1)
a2 = 1/2 * (a2 + a2.T)

def legendreQ(l, xi, xi_1):
    return sp.special.lqn(l, 1 + 4 * (1 + np.cos(xi)) / ((1 - np.cos(xi))*(1 + np.cos(xi_1))))

def PhiHat_l(l, xi, xi_1):
    return np.sin(xi_1) * ( (l == 0) * (-2/(1 + np.cos(xi_1))) + 8*(1 + np.cos(xi)) / ((1 + np.cos(xi_1))^2 * (1 - np.cos(xi))) * legendreQ(l, xi, xi_1) )

av1 = 1 + 4 * (1 + np.cos(xi)) / ( (1 + np.cos(xi_1)) * (1 - np.cos(xi)) )
bv1 = 1 + 4 * (1 + np.cos(xi)) / ( (1 + np.cos(xi_2)) * (1 - np.cos(xi)) )


legendreQMatrix = np.zeros((2*M, 2*M, 2*M))
print("legendreQMatrix:", legendreQMatrix)
