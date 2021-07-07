import numpy as np
np.random.seed(1)
image=np.random.randn(1, 256, 256, 3) #h256X256 image
W1=np.random.randn(3, 3, 3, 32)
b1=np.random.randn(1, 1, 1, 32)
W2=np.random.randn(3, 3, 32, 64)
b2=np.random.randn(1, 1, 1, 64)
W3=np.random.randn(3, 3, 64, 128)
b3=np.random.randn(1, 1, 1, 128)
hparameters1 = {"pad" : 129,"stride": 2}
hparameters3 = {"pad" : 65,"stride": 2}
hparameters2 = {"stride" : 2, "f": 2}
hparameters4 = {"stride" : 2, "f": 2}
hparameters5 = {"pad" : 32,"stride": 2}
