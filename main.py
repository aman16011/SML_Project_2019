import numpy as np
from glob import glob

class MotorImageryDataset:
    def __init__(self, dataset='./data/A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)

        self.Fs = 250 # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=3):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []
        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trials.append(trial)

            except:
                continue

        return trials, classes

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

X,Y = get_data("./data/")
print(len(X),len(Y),len(X[0]))
