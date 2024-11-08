import numpy as np

# The dual has B0h, B1h, and B2h as extra parameters to the primal
dual = True

print("\n")

M = 40
lmax = 20
print("M:", M)
print("lmax:", lmax)

Mfn = int(M * (M + 1) / 2)
Mfl = (lmax + 1) * M
print("Mfn:", Mfn)
print("Mfl:", Mfl)

xi = M

A0h = 1
B0h = 1 if (dual == True) else 0
hl0_RM = Mfl

A1h = M
B1h = M if (dual == True) else 0
hl1_RM = Mfl * M
hl1_IM = Mfl * M

A2h = Mfn
B2h = Mfn if (dual == True) else 0
hl2_RM = Mfl * Mfn
hl2_IM = Mfl * Mfn

hl0_RMA = Mfl
hl1_IMA = Mfl * M
hl1_RMA = Mfl * M
hl2_IMA = Mfl * Mfn
hl2_RMA = Mfl * Mfn
hl1_IMB = Mfl * M
hl2_IMB = Mfl * Mfn

LambdaV = Mfl


print("########################################")
print("########## POINT DISTRIBUTION ##########")
print("########################################")
print("xi:", xi)
print("A0h:", A0h)
print("B0h:", B0h)
print("hl0_RM:", hl0_RM)
print("A1h:", A1h)
print("B1h:", B1h)
print("hl1_RM:", hl1_RM)
print("hl1_IM:", hl1_IM)
print("A2h:", A2h)
print("B2h:", B2h)
print("hl2_RM:", hl2_RM)
print("hl2_IM:", hl2_IM)
print("hl0_RMA:", hl0_RMA)
print("hl1_IMA:", hl1_IMA)
print("hl1_RMA:", hl1_RMA)
print("hl2_IMA:", hl2_IMA)
print("hl2_RMA:", hl2_RMA)
print("hl1_IMB:", hl1_IMB)
print("hl2_IMB:", hl2_IMB)
print("LambdaV:", LambdaV)




total = [xi, A0h, B0h, hl0_RM, A1h, B1h, hl1_RM, hl1_IM, A2h, B2h, hl2_RM, hl2_IM, hl0_RMA, hl1_IMA, hl1_RMA, hl2_IMA, hl2_RMA, hl1_IMB, hl2_IMB, LambdaV]
cum = np.cumsum(total)

print("#############################################")
print("########## CUMULATIVE DISTRIBUTION ##########")
print("#############################################")
print(f"xi: (0, {cum[0]}]")
print(f"A0h ({cum[0]},  {cum[1]}]")
print(f"B0h ({cum[1]},  {cum[2]}]")
print(f"hl0_RM ({cum[2]},  {cum[3]}]")
print(f"A1h ({cum[3]},  {cum[4]}]")
print(f"B1h ({cum[4]},  {cum[5]}]")
print(f"hl1_RM ({cum[5]},  {cum[6]}]")
print(f"hl1_IM ({cum[6]},  {cum[7]}]")
print(f"A2h ({cum[7]},  {cum[8]}]")
print(f"B2h ({cum[8]},  {cum[9]}]")
print(f"hl2_RM ({cum[9]},  {cum[10]}]")
print(f"hl2_IM ({cum[10]},  {cum[11]}]")
print(f"hl0_RMA ({cum[11]},  {cum[12]}]")
print(f"hl1_IMA ({cum[12]},  {cum[13]}]")
print(f"hl1_RMA ({cum[13]},  {cum[14]}]")
print(f"hl2_IMA ({cum[14]},  {cum[15]}]")
print(f"hl2_RMA ({cum[15]},  {cum[16]}]")
print(f"hl1_IMB ({cum[16]},  {cum[17]}]")
print(f"hl2_IMB ({cum[17]},  {cum[18]}]")
print(f"LambdaV ({cum[18]},  {cum[19]}]")

print(len(total))