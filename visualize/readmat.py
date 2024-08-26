import scipy.io
import numpy as np
import mat73
import os

file_path = (
    "/Users/wengweishan/Documents/MATLAB/Ultrasound/output/0821_inhomo_analyt.mat"
)
saveanme = os.path.basename(file_path).split(".")[0] + ".npz"

mat = mat73.loadmat(file_path)["sr_trunc"]
print(mat.shape)
mat = mat.reshape(500, 500, -1)
mat = np.transpose(mat, (2, 0, 1))
print(mat.shape)
np.savez_compressed(saveanme, data=mat)
