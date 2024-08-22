import os
import numpy as np
# import scipy as sp
from scipy.special import lqn
# import sympy as smp
import cvxpy as cp
# import matplotlib.pyplot as plt
# import mpmath as mp



np.set_printoptions(precision = 600)

########## VALUES ##########
# V = [[-1, 40, 20], [0, 40, 20], [4/3, 40, 20], [2, 40, 20], [3, 40, 20], [3.8, 40, 20]]
V = [4/3, 2, 3, 3.8]
for i in V:
    sa = i                                     # [?]
    M = 40                                                              # Number of interpolation points
    lmax = 20                                                           # Maximum angular momentum

    # xi = np.pi/M * (np.arange(1, 2*M + 1) - 1/2)                      # Method 1
    # xi = [np.pi/M * (j - 1/2) for j in range(1, 2*M+1)]               # Method 2
    xi = np.linspace(np.pi/M * (1/2), np.pi/M * (2*M - 1/2), 2*M)       # Interpolation points
    s = sa + (4 - sa) / np.cos(xi/2)**2                                 # Mapping from xi to s
    rho = np.pi/4 * np.sqrt((s - 4)/s)                                  # Mapping from s to rho

    # UNUSED
    # def zf(s):
    #     return (np.sqrt(4 - sa) - np.sqrt(4 - s)) / (np.sqrt(4 - sa) + np.sqrt(4 - s))

    print("sa =", sa)
    print("M =", M)
    print("lmax =", lmax)

    # print("xi = ", xi)
    # print("s = ", s) # algebraic type sa?
    # print("rho = ", rho) # algebraic type sa?


    def KHat(M, j1, j2):
        return 1/(2*M) * (1 - (-1)**(j1 - j2)) * 1/np.tan(np.pi/(2*M) * (j1 - j2))

    KMatrix = np.zeros((2*M, 2*M))
    print(np.shape(KMatrix))
    for j1 in range(1, 2*M + 1):
        for j2 in range(j1 + 1, 2*M + 1):
            KMatrix[j1 - 1, j2 - 1] = KHat(M, j1, j2)
            KMatrix[j2 - 1, j1 - 1] = - KMatrix[j1 - 1, j2 - 1]

            np.nan_to_num(KMatrix, copy = False)

    print(KMatrix)




    KMatrix_i = np.zeros((M, M))
    KMatrix_d = np.zeros((M, M))
    for j1 in range(1, M + 1):
        for j2 in range(1, M + 1):
            KMatrix_i[j1 - 1, j2 - 1] = - (KMatrix[j1 - 1, j2 - 1] - KMatrix[j1 - 1, 2*M - j2])
            KMatrix_d[j1 - 1, j2 - 1] = KMatrix[j1 - 1, j2 - 1] + KMatrix[j1 - 1, 2*M - j2]

    fvra = np.cos(4*xi)
    fvia = np.sin(4*xi)
    fviaN = np.dot(KMatrix, fvra)
    norm1 = np.linalg.norm(fvia - fviaN)
    print("norm1:", norm1)

    fvr = np.cos(4*xi)[:M]
    fvi = np.sin(4*xi)[:M]
    fvrN = np.dot(KMatrix_i, fvi)
    fviN = np.dot(KMatrix_d, fvr)
    norm2 = np.linalg.norm(fvr - fvrN)
    norm3 = np.linalg.norm(fvi - fviN)
    print("norm2:", norm2)
    print("norm3:", norm3)



    s0, t0, u0 = 4/3, 4/3, 4/3

    def mathcalK(xi, s):
        return 1/np.pi * np.sin(xi) / (8/s - 1 - np.cos(xi))

    def Kappa(x1, s):
        return 1/np.pi * np.sqrt((x1 - 4)/(4 - sa)) * (s - sa)/(x1 - s)

    as1 = Kappa(s, s0)
    at1 = Kappa(s, t0)
    au1 = Kappa(s, u0)

    a0 = 1
    a1 = as1 + at1 + au1
    a2 = np.outer(as1, at1) + np.outer(as1, au1) + np.outer(at1, au1)
    a2 = 1/2 * (a2 + a2.T)




    def Lambda(l, s):
        return ( (np.sqrt(s) - 2)/(np.sqrt(s) + 2) ) ** (l/2)

    LambdaMatrix = np.ones((lmax + 1, M))
    print(np.shape(LambdaMatrix))
    for l in range(0, lmax + 1):
        for j1 in range(1, M + 1):
            LambdaMatrix[l, j1 - 1] = Lambda(2*l, s[j1 - 1])
            np.nan_to_num(LambdaMatrix, copy = False)

    print(LambdaMatrix)




    def LegendreQ(l, x, s):
        return lqn(l, 1 + 2*x/(s - 4))

    legendreQMatrix = np.zeros((lmax + 1, M, M))
    print(np.shape(legendreQMatrix))
    for l in range(0, lmax + 1):
        for j1 in range(1, M + 1):
            for j in range(1, M + 1):
                legendreQMatrix[l, j1 - 1, j - 1] = LegendreQ(2*(l + 1), s[j1 - 1], s[j - 1])[0][2*l]
                np.nan_to_num(legendreQMatrix, copy = False)                  # Replace NaN with 0

                # print("l:", l)
                # print("j1:", j1)
                # print("j:", j)
                # print("leg:", LegendreQ(2*(l + 1), s[j1 - 1], s[j - 1])[0][2*l])

    print(legendreQMatrix)




    def PhiHat(l, x, s, Q):
        # if (l == 0):
        return 1/np.pi * np.sqrt((x - 4)/(4 - sa)) * ((l == 0)*(-2) + 4*(x - sa)/(s - 4) * Q)
        # else:
        #     return 1/np.pi * np.sqrt((x - 4)/(4 - sa)) * (4*(x - sa)/(s - 4) * Q)

    PhiHatMatrix = np.zeros((lmax + 1, M, M))
    print(np.shape(PhiHatMatrix))
    for l in range(0, lmax + 1):
        for j1 in range(1, M + 1):
            for j in range(1, M + 1):
                PhiHatMatrix[l, j1 - 1, j - 1] = PhiHat(l, s[j1 - 1], s[j - 1], legendreQMatrix[l, j1 - 1, j - 1])
                np.nan_to_num(PhiHatMatrix, copy = False)                  # Replace NaN with 0

                # print("l:", l)
                # print("j1:", j1)
                # print("j:", j)
                # print("PhiHat:", PhiHat(l, s[j1 - 1], s[j - 1]) * legendreQMatrix[l, j1 - 1, j - 1])

    print(PhiHatMatrix)





    def PhiTilde(l, x1, x2, s, Q1, Q2):
        return 1/(np.pi)**2 * np.sqrt((x1 - 4) * (x2 - 4))/(4 - sa) * ((l == 0) * 2 - 4*(x1 - sa) * (s - 4 + sa + x1)/(s - 4 + x1 + x2)/(s - 4) * Q1 - 4 * (x2 - sa) * (s - 4 + sa + x2)/(s - 4 + x1 + x2)/(s - 4) * (-1)**l * Q2)

    PhiTildeMatrix = np.zeros((lmax + 1, M, M, M))
    print(np.shape(PhiTildeMatrix))
    for l in range(0, lmax + 1):
        for j in range(1, M + 1):
            for j1 in range(1, M + 1):
                for j2 in range(j1, M + 1):
                    PhiTildeMatrix[l, j1 - 1, j2 - 1, j - 1] = PhiTilde(l, s[j1 - 1], s[j2 - 1], s[j - 1], legendreQMatrix[l, j1 - 1, j - 1], legendreQMatrix[l, j2 - 1, j - 1])
                    PhiTildeMatrix[l, j2 - 1, j1 - 1, j - 1] = PhiTildeMatrix[l, j1 - 1, j2 - 1, j - 1]
                    np.nan_to_num(PhiTildeMatrix, copy = False)                  # Replace NaN with 0

                    # print("l:", l)
                    # print("j1:", j1)
                    # print("j2:", j2)
                    # print("PhiTilde:", PhiTilde(l, xi[j1 - 1], xi[j2 - 1]))

    # print(PhiTildeMatrix)










    Delta = np.pi / M

    def delta(x, y):
        return 1 if x == y else 0

    Mfn = int(M * (M + 1) / 2)
    Mfl = (lmax + 1) * M                            # Number of interpolation points in the partial waves
    A0h = 1                                         # f0    -> functional
    hl0_RM = np.zeros((Mfl))                        # f0    -> Re(h)
    A1h = np.zeros(M)                               # sigma -> functional
    hl1_RM = np.zeros((Mfl, M))                     # sigma -> Re(h)
    hl1_IM = np.zeros((Mfl, M))                     # sigma -> Im(h)
    A2h = np.zeros(Mfn)                             # rho   -> functional
    hl2_RM = np.zeros((Mfl, Mfn))                   # rho   -> Re(h)
    hl2_IM = np.zeros((Mfl, Mfn))                   # rho   -> Im(h)

    print("Mfn:", Mfn)
    print("Mfl:", Mfl)

    # Now we put computation data in matrix form
    ##### F0 #####
    for j in range(1, M + 1):
        jf1 = j - 1  # only l = 0
        hl0_RM[jf1] = 2 * rho[j - 1]

    ##### SIGMA #####
    for j in range(1, M + 1):
        A1h[j - 1] = a1[j - 1] * Delta

    for j in range(1, M + 1):
        hl1_IM[j - 1, j - 1] = 2 * rho[j - 1]

    for j1 in range(1, M + 1):  # l = 0 part
        for j in range(1, M + 1):
            hl1_RM[j - 1, j1 - 1] = 2 * rho[j - 1] * KMatrix_i[j - 1, j1 - 1]

    for l in range(0, lmax + 1):
        for j in range(1, M + 1):
            for j1 in range(1, M + 1):
                jf1 = l * M + j - 1
                jf2 = j1 - 1
                hl1_RM[jf1, jf2] += 2 * rho[j - 1] * Delta * PhiHatMatrix[l, j1 - 1, j - 1]

    ##### RHO #####
    jf2 = 0
    for j1 in range(1, M + 1):
        for j2 in range(j1, M + 1):
            A2h[jf2] = a2[j1 - 1, j2 - 1] * Delta**2
            jf2 += 1

    for l in range(0, lmax + 1):
        for j in range(1, M + 1):
            jf2 = 0
            for j1 in range(1, M + 1):
                for j2 in range(j1, M + 1):
                    jf1 = l * M + j - 1
                    hl2_RM[jf1, jf2] = rho[j - 1] * (KMatrix_i[j - 1, j1 - 1] * Delta * PhiHatMatrix[l, j2 - 1, j - 1] + KMatrix_i[j - 1, j2 - 1] * Delta * PhiHatMatrix[l, j1 - 1, j - 1] + Delta**2 * PhiTildeMatrix[l, j1 - 1, j2 - 1, j - 1])
                    jf2 += 1

    for l in range(0, lmax + 1):
        for j in range(1, M + 1):
            jf2 = 0
            for j1 in range(1, M + 1):
                for j2 in range(j1, M + 1):
                    jf1 = l * M + j - 1
                    hl2_IM[jf1, jf2] = rho[j - 1] * (delta(j - 1, j1 - 1) * Delta * PhiHatMatrix[l, j2 - 1, j - 1] + delta(j - 1, j2 - 1) * Delta * PhiHatMatrix[l, j1 - 1, j - 1])
                    jf2 += 1

    print("Saving coefficients...")
    print("Done")






    hl0_RMA = np.zeros(Mfl)                         # f0    -> Re(h/Lambda)
    hl1_IMA = np.zeros((Mfl, M))                    # sigma -> Im(h/Lambda)
    hl1_RMA = np.zeros((Mfl, M))                    # sigma -> Re(h/Lambda)
    hl2_IMA = np.zeros((Mfl, Mfn))                  # rho   -> Im(h/Lambda)
    hl2_RMA = np.zeros((Mfl, Mfn))                  # rho   -> Re(h/Lambda)
    hl1_IMB = np.zeros((Mfl, M))                    # sigma -> Im(h/Lambda^2)
    hl2_IMB = np.zeros((Mfl, Mfn))                  # rho   -> Im(h/Lambda^2)


    LambdaV = np.ones(Mfl)
    for l in range(0, lmax + 1):
        for j in range(1, M + 1):
            jf1 = l * M + j - 1
            LambdaV[jf1] = LambdaMatrix[l, j - 1]

    for jf1 in range(1, Mfl + 1):
        hl0_RMA[jf1 - 1] = hl0_RM[jf1 - 1] / LambdaV[jf1 - 1]

    for jf1 in range(1, Mfl + 1):
        for jf2 in range(0, M):
            hl1_RMA[jf1 - 1, jf2 - 1] = hl1_RM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]
            hl1_IMA[jf1 - 1, jf2 - 1] = hl1_IM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]
            hl1_IMB[jf1 - 1, jf2 - 1] = hl1_IM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]**2

    for jf1 in range(1, Mfl + 1):
        for jf2 in range(0, Mfn):
            hl2_RMA[jf1 - 1, jf2 - 1] = hl2_RM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]
            hl2_IMA[jf1 - 1, jf2 - 1] = hl2_IM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]
            hl2_IMB[jf1 - 1, jf2 - 1] = hl2_IM[jf1 - 1, jf2 - 1] / LambdaV[jf1 - 1]**2





    saf = str(sa).replace(".", "p")
    dataKFile = os.path.join(os.getcwd(), f"SmQCD_primalK{saf}_points_data_P2_nopole_lmax{lmax}_M{M}.dat")

    with open(dataKFile, "w") as fd:
        for j in range(1, M + 1):
            fd.write(f"{xi[j - 1]}\n")

        fd.write(f"{A0h}\n")

        for jf1 in range(1, Mfl + 1):
            fd.write(f"{hl0_RM[jf1 - 1]}\n")

        for j in range(1, M + 1):
            fd.write(f"{A1h[j - 1]}\n")

        for jf2 in range(1, M + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl1_RM[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, M + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl1_IM[jf1 - 1, jf2 - 1]}\n")

        for j in range(1, Mfn + 1):
            fd.write(f"{A2h[j - 1]}\n")

        for jf2 in range(1, Mfn + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl2_RM[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, Mfn + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl2_IM[jf1 - 1, jf2 - 1]}\n")

        for jf1 in range(1, Mfl + 1):
            fd.write(f"{hl0_RMA[jf1 - 1]}\n")

        for jf2 in range(1, M + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl1_RMA[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, M + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl1_IMA[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, Mfn + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl2_RMA[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, Mfn + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl2_IMA[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, M + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl1_IMB[jf1 - 1, jf2 - 1]}\n")

        for jf2 in range(1, Mfn + 1):
            for jf1 in range(1, Mfl + 1):
                fd.write(f"{hl2_IMB[jf1 - 1, jf2 - 1]}\n")

        for jf1 in range(1, Mfl + 1):
            fd.write(f"{LambdaV[jf1 - 1]}\n")