import numpy as np
from glob import glob
from helper_function import *
from scipy.signal import welch
from sklearn.metrics import accuracy_score

def preprocess(X):
    X_filtered = X.copy()
    for i in range(len(X[0])):
        for j in range(22):
            X_filtered[i,j,:] = butter_bandpass_filter(X[i,j],8,24,250)
       
    return X_filtered

# Visualization of filtered signal - how only one frequency band (8-24Hz) remains now.

def feature_extraction(X_filtered):
    # Feature extraction

        # Average Bandpower features [Mu and Beta band power features 8-24Hz with 2Hz binning- 8 bins per channel]
    X_tr = np.zeros((len(X_filtered),22*8))   
    for i in range(len(X_filtered)):
        dum = np.zeros((22,8))
        for ch in range(22):
            freqs, psd_vals = welch(X_filtered[i,ch,:],250)
            dum[ch,0] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=8,freqs<=10))])
            dum[ch,1] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=10,freqs<=12))])
            dum[ch,2] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=12,freqs<=14))])
            dum[ch,3] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=14,freqs<=16))])
            dum[ch,4] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=16,freqs<=18))])
            dum[ch,5] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=18,freqs<=20))])
            dum[ch,6] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=20,freqs<=22))])
            dum[ch,7] = np.sum(psd_vals[np.argwhere(np.logical_and(freqs>=22,freqs<=24))])        
        X_tr[i] = dum.reshape(1,22*8)
    return X_tr

def get_result(X_train,Y_train,X_val,Y_val):
    X_train_pca,X_val_pca = pca(X_train,X_val)
    X_train_lda,X_val_lda = lda(X_train,Y_train,X_val)
    X_train_ica,X_val_ica = ICA(X_train,Y_train,X_val)      
    
    ### LOGISTIC REGRESSION
    # PCA results
    model_pca = learn_LR_classifier(X_train_pca,Y_train)
    Y_train_pred = model_pca.predict(X_train_pca)
    print("Train accuracy of PCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of PCA",accuracy_score(model_pca.predict(X_val_pca),Y_val))

    # LDA results 
    model_lda = learn_LR_classifier(X_train_lda,Y_train)
    Y_train_pred = model_lda.predict(X_train_lda)
    print("Train accuracy of LDA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of LDA",accuracy_score(model_lda.predict(X_val_lda),Y_val))

    # ICA results
    model_ica = learn_LR_classifier(X_train_ica,Y_train)
    Y_train_pred = model_ica.predict(X_train_ica)
    print("Train accuracy of ICA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of ICA",accuracy_score(model_ica.predict(X_val_ica),Y_val))

    # LDA over PCA
    X_train_lda,X_val_lda = lda(X_train_pca,Y_train,X_val_pca)
    model_lda = learn_LR_classifier(X_train_lda,Y_train)
    Y_train_pred = model_lda.predict(X_train_lda)
    print("Train accuracy of LDA over PCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of LDA over PCA",accuracy_score(model_lda.predict(X_val_lda),Y_val))

    ### NAIVE BAYES
    # PCA results
    
    model_pca = learn_naive_bayes_classifier(X_train_pca,Y_train)
    Y_train_pred = model_pca.predict(X_train_pca)
    print("Train accuracy of PCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of PCA",accuracy_score(model_pca.predict(X_val_pca),Y_val))

    # LDA results 
    model_lda = learn_naive_bayes_classifier(X_train_lda,Y_train)
    Y_train_pred = model_lda.predict(X_train_lda)
    print("Train accuracy of LDA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of LDA",accuracy_score(model_lda.predict(X_val_lda),Y_val))

    # ICA results
    model_ica = learn_naive_bayes_classifier(X_train_ica,Y_train)
    Y_train_pred = model_ica.predict(X_train_ica)
    print("Train accuracy of ICA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of ICA",accuracy_score(model_ica.predict(X_val_ica),Y_val))

    # LDA over PCA
    X_train_lda,X_val_lda = lda(X_train_pca,Y_train,X_val_pca)
    model_lda = learn_naive_bayes_classifier(X_train_lda,Y_train)
    Y_train_pred = model_lda.predict(X_train_lda)
    print("Train accuracy of LDA over PCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of LDA over PCA",accuracy_score(model_lda.predict(X_val_lda),Y_val))

def get_k_fold_result(X_train,Y_train,X_val,Y_val):

    for l in range(len(X_train)):
        get_result(X_train[l],Y_train[l],X_val[l],Y_val[l])
        quit(0)
#USE LOGSCALE IF THE DIFFERENCE IN VALUES ISN'T THAT BIG ??

# Visualization



# Dimensionality reduction
    #ICA
    #PCA
    #LDA

# Visualization

# Classification
    

# Analysis

if __name__ == '__main__':
    # # Assigning class labels
    labels = {'tongue':0,'foot':1, 'left':2, 'right':3}
    # Dataset extraction
    X,Y = get_data("../data/") # trials X channels X values
    for l in range(len(Y)):
        Y[l] = labels[Y[l]]
    X = preprocess(X)
    X = feature_extraction(X)
    print(X.shape,Y.shape)
    np.save("X_train.npy",X)
    np.save("Y_train.npy",Y)
    quit(0)
    X = np.load("X_train.npy")
    Y = np.load("Y_train.npy")
    split = 2
    # K- Fold Split
    
    X_train,Y_train,X_val,Y_val = stratified_K_fold(split,X,Y)
    print("splitting done")

    # Results
    get_k_fold_result(X_train,Y_train,X_val,Y_val)
