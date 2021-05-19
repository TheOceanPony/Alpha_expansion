import copy
import numpy as np

from numpy.random import shuffle
import matplotlib.pyplot as plt

import funcs as f
import utils as utl

from time import time
start = time()


C = np.array([1, 128, 255])
repeats = 2
img = utl.import_img("input.png", bw=True, newshape=(128,128))


#Main
labeling = f.initial_labeling(img, C)

for iteration in range(repeats):
    a_shuffle = np.copy(C)
    shuffle(a_shuffle)
    
    for i in range(C.size):

        # 1) alpha_i
        a_i = a_shuffle[i]

        # 2) initial labeling
        k_init = labeling

        # 3) init new graph
        g = f.init_g(img, k_init, a_i, scale=50)

        # 4) solve for new binary task
        res = f.Ford_Falkerson(img, g)
        labeling = f.translate_to_labeling(res, k_init, a_i)
        
        
print(f"Time: {time() - start}")
plt.subplots(figsize=(10, 10))
plt.imsave("res.png", np.reshape(labeling, (128,128)), cmap='gray')
