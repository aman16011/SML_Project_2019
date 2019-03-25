# all functions go here.
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import glob

# Data Loader
def get_data(folder_path):

    files = glob(folder_path+"*")
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
            trials, classes = ImageryDataset.get_trials_from_channel()
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