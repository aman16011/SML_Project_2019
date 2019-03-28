# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:42:01 2019

@author: Aman Roy
"""

# all functions go here.
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import glob
from scipy.signal import butter,lfilter
from MIclass import MotorImageryDataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA 
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

# Data Loader
def get_data(folder_path):

    files = glob.glob(folder_path+"*")
    X = []
    Y = []
    for file in files:
        print(file)
        ImageryDataset = MotorImageryDataset(file)
        trials, classes = ImageryDataset.get_trials_from_channel(1)
        X_temp = trials
        Y_temp = classes
        print(len(X_temp),len(X_temp[0]),len(Y_temp))
        for i in range(2,23):
            trials, classes = ImageryDataset.get_trials_from_channel(i)
            X_temp = np.concatenate([X_temp,trials],axis = 1)
        print(len(X_temp),len(X_temp[0]),len(Y_temp))
        if len(X) == 0:
            X = X_temp
            Y = Y_temp
        else:
            X = np.append(X,X_temp,0)
            Y = np.append(Y,Y_temp)

    return X,Y

# Logistic Regression Classifier.
def learn_LR_classifier(X_train,Y_train):

    return LogisticRegression().fit(X_train,Y_train)

# Gaussian Naive Bayes Classifier.
def learn_naive_bayes_classifier(X_train,Y_train):

    return GaussianNB().fit(X_train,Y_train)

# does k_fold_splitting and is used for CV.
def stratified_K_fold(split,X,Y):
    
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    skf = StratifiedKFold(n_splits=split)
    for train_index, val_index in skf.split(X,Y):
        X_train_temp, X_val_temp = X[train_index], X[val_index]
        Y_train_temp, Y_val_temp = Y[train_index], Y[val_index]
        X_train.append(X_train_temp)
        Y_train.append(Y_train_temp)
        X_val.append(X_val_temp)
        Y_val.append(Y_val_temp)
        print(len(X_train_temp),"Training size",len(X_val_temp),"validation size")

    return X_train,Y_train,X_val,Y_val

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def pca(X_train,X_test):
    
    clf = PCA(n_components = 20)
    clf.fit(X_train)
    X_transform_train = clf.transform(X_train)
    X_transform_test = clf.transform(X_test)

    return X_transform_train,X_transform_test

def lda(X_train,Y_train,X_test):

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train,Y_train)
    X_transform_train = clf.transform(X_train)
    X_transform_test = clf.transform(X_test)

    return X_transform_train,X_transform_test    

def KPCA(X_train,X_test):
    transformer = KernelPCA(kernel='rbf')
    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.fit_transform(X_test)
    return X_train_trans, X_test_trans    

def preprocess(X):
    X_filtered = X.copy()
    for i in range(len(X[0])):
        for j in range(22):
            X_filtered[i,j,:] = butter_bandpass_filter(X[i,j],8,24,250)
       
    return X_filtered


def feature_extraction(X_filtered):
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


    # LDA over PCA
    X_train_lda,X_val_lda = lda(X_train_pca,Y_train,X_val_pca)
    model_lda = learn_LR_classifier(X_train_lda,Y_train)
    Y_train_pred = model_lda.predict(X_train_lda)
    print("Train accuracy of LDA over PCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of LDA over PCA",accuracy_score(model_lda.predict(X_val_lda),Y_val))
    
    #KPCA
    model_pca = learn_LR_classifier(X_train_KPCA,Y_train)
    Y_train_pred = model_pca.predict(X_train_KPCA)
    print("Train accuracy of KPCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of KPCA",accuracy_score(model_pca.predict(X_val_KPCA),Y_val))


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
    
    # Kernel PCA
    X_train_KPCA,X_val_KPCA = KPCA(X_train, X_val)
    model_KPCA = learn_naive_bayes_classifier(X_train_KPCA,Y_train)
    Y_train_pred = model_lda.predict(X_train_KPCA)
    print("Train accuracy of KPCA",accuracy_score(Y_train_pred,Y_train)) 
    print("Val accuracy of KPCA",accuracy_score(model_KPCA.predict(X_val_KPCA),Y_val))
    
    
def get_k_fold_result(X_train,Y_train,X_val,Y_val):

    for l in range(len(X_train)):
        get_result(X_train[l],Y_train[l],X_val[l],Y_val[l])
        quit(0)


