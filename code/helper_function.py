# all functions go here.
import glob

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
