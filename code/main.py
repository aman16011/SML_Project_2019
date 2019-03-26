import numpy as np
from glob import glob
from helper_function import *
from scipy.signal import welch

# Dataset extraction
X,Y = get_data("./data/") # trials X channels X values
X_filtered = X.copy()
#print(len(X),len(Y),len(X[0]))

# Visualization

# Dataset Pre-processing

    # Notch Filtering
    # EOG artifact removal
    # Butterworth BandPass Filtering
for i in range(len(X[0])):
    for j in range(22):
        X_filtered[i,j,:] = butter_bandpass_filter(X[i,j],8,24,250)        

# Visualization of filtered signal - how only one frequency band (8-24Hz) remains now.

# Feature extraction

    # Average Bandpower features [Mu and Beta band power features 8-24Hz with 2Hz binning- 8 bins per channel]
X_tr = np.zeros((len(X_filtered),22*8))   
for i in range(len(X_filtered)):
    dum = np.zeros((22,8))
    for ch in range(22):
        freqs, psd_vals = welch(X_filtered[i,ch,:],250)
        dum[ch,0] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=8,freqs<=10))])
        dum[ch,1] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=10,freqs<=12))])
        dum[ch,2] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=12,freqs<=14))])
        dum[ch,3] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=14,freqs<=16))])
        dum[ch,4] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=16,freqs<=18))])
        dum[ch,5] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=18,freqs<=20))])
        dum[ch,6] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=20,freqs<=22))])
        dum[ch,7] = np.mean(psd_vals[np.argwhere(np.logical_and(freqs>=22,freqs<=24))])        
    X_tr[i] = dum.reshape(1,22*8)
    
#USE LOGSCALE IF THE DIFFERENCE IN VALUES ISN'T THAT BIG ??

# Visualization


# Dimensionality reduction
    #ICA
    #PCA
    #LDA

# Visualization

# Classification
    

# Analysis
