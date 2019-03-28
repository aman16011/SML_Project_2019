import numpy as np
from glob import glob
from helper_function import *
from scipy.signal import welch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
# # Assigning class labels
labels = {'tongue':0,'foot':1, 'left':2, 'right':3}
# Dataset extraction
X,Y = get_data("../data/") # trials X channels X values

#Showing PSD of 1st subject 1st trial all 22 channels: 
for i in range(22):
    f,psd = welch(X)
    plt.plot(f,psd)
    plt.savefig('PSD before filtering')
    
for l in range(len(Y)):
    Y[l] = labels[Y[l]]

# Pre-processing
X = preprocess(X)

# Visualization of filtered signal - how only one frequency band (8-24Hz) remains now.
for i in range(22):
    f,psd = welch(X[0,i,:])
    plt.plot(f,psd)
    plt.savefig('PSD before filtering')

# Feature extraction
# Average Bandpower features [Mu and Beta band power features 8-24Hz with 2Hz binning- 8 bins per channel]

X = feature_extraction(X)
print(X.shape,Y.shape)
np.save("X_train.npy",X)
np.save("Y_train.npy",Y)

#??? Class conditional density visualizations
#??? Reduced Dimension Visualization

X = np.load("X_train.npy")
Y = np.load("Y_train.npy")
split = 2
# K- Fold Split

X_train,Y_train,X_val,Y_val = stratified_K_fold(split,X,Y)
print("splitting done")

# Results
get_k_fold_result(X_train,Y_train,X_val,Y_val)

    
