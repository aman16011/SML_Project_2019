import numpy as np
from glob import glob
from helper_function import *
from scipy.signal import welch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import pickle as pkl

# # Assigning class labels
labels = {'tongue':0,'foot':1, 'left':2, 'right':3}
# Dataset extraction
X_original,Y = get_data("../data/") # trials X channels X values

#Create Chunks
X = np.zeros((len(X_original)*3,22,250))    
count1=0
count2=0
count3=0
for tr in range(len(X_original)):
    for ch in range(22):
        X[count1,ch,:] = X_original[tr,ch,750:1000]
        X[count2,ch,:] = X_original[tr,ch,1000:1250]        
        X[count3,ch,:] = X_original[tr,ch,1250:1500]
    count1+=1
    count2+=1
    count3+=1

pkl.dump(X,open(r'X.pkl','wb'))
        
#Showing PSD of 1st subject 1st trial all 22 channels: 
for i in range(22):
    f,psd = welch(X[0,i,:],250)
    plt.plot(f,psd)
    plt.savefig('PSD before filtering.png')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('PSD')
    plt.title('Power spectral density (Before filtering) for subject 1 trial 1')    
for l in range(len(Y)):
    Y[l] = labels[Y[l]]


# Pre-processing
X = preprocess(X)

# Visualization of filtered signal - how only one frequency band (8-24Hz) remains now.
for i in range(22):
    f,psd = welch(X[0,i,:],250)
    plt.plot(f,psd)
    plt.savefig('PSD after filtering.png')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('PSD')
    plt.title('Power spectral density (After filtering) for subject 1 trial 1')    
    plt.savefig('PSD after filtering.png')

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

    
