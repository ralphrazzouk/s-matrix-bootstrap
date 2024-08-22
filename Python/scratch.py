import os
import numpy as np
import pandas as pd

testFile = os.path.join(os.getcwd(), f"Python/test.txt")
print(testFile)
test = np.zeros((16, 2))
with open(testFile, 'r') as testID:
    for i in range(0, 2):
        for j in range(0, 16):
            test[j, i] = float(testID.readline().split()[0])

print(pd.DataFrame(test))