import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

import json
with open("../datas/dataset/classes.json", "r") as f:
    d = json.load(f)
    
    classes = {}
    for word in d:
        classes[word] = len(classes) + 1


def saveData(data, name, prefix = "../datas/demo dataset flipped/",file_format = "pkl"):
    path = prefix + f"{name}.{file_format}"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    

def loadData(name, prefix = "../datas/demo dataset flipped/", file_format = "pkl"):
    path = prefix + f"{name}.{file_format}"
    with open(path, "rb") as f:
        return pickle.load(f)


def padSample(sample, target_lenght = 40):
    if sample.shape[0] < target_lenght:
        difference = target_lenght - sample.shape[0]
        
        padding = np.zeros((difference, sample.shape[1]))
        
        padded_array = np.vstack((padding, sample))
        
        return padded_array
    return sample


def shiftSample(sample):
    for i, frame in enumerate(sample):
        frame = frame.reshape(-1, 2)
        x_values = frame[ : , 0]
        y_values = frame[ : , 1]
        if np.max(x_values) != 0 and np.max(y_values) != 0:
            min_x = np.min(x_values[x_values > 0])
            min_y = np.min(y_values[y_values > 0])
            
            x_values = np.clip(x_values - min_x, 0, None)
            y_values = np.clip(y_values - min_y, 0, None)
        
        coords_subtracted = np.column_stack((x_values, y_values)).flatten()
        
        sample[i] = coords_subtracted
    
    return sample 

def flatten(frame):
    hands = frame[0]
    while len(hands) < 2:
        hands.append([[0, 0] for _ in range(21)])
    hand1Coords = [coord for pair in hands[0] for coord in pair]
    hand2Coords = [coord for pair in hands[1] for coord in pair]
    headCoords = list(frame[1])
    flattened_frame = hand1Coords + hand2Coords + headCoords
    
    return np.array(flattened_frame)


def processSamples(X, datatype = "train"):
    for i, sample in enumerate(X):
        sample = np.array([flatten(frame) for frame in sample])
        sample = shiftSample(sample)
        sample = padSample(sample)
        X[i] = sample
    
    X = np.array(X)
    
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    if datatype == "train":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        saveData(scaler, "scaler")
    
    else:
        scaler = loadData("scaler")
        X_scaled = scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
    
    return X_scaled


X_train = loadData("X_train")
X_train = processSamples(X_train, datatype="train")
saveData(X_train, "X_train_final")


X_test = loadData("X_test")
X_test = processSamples(X_test, datatype="test")
saveData(X_test, "X_test_final")


X_val = loadData("X_val")
X_val = processSamples(X_val, datatype="val")
saveData(X_val, "X_val_final")


y_train = np.array([classes[text] for text in loadData("y_train")])
saveData(y_train, "y_train_final")


y_test = np.array([classes[text] for text in loadData("y_test")])
saveData(y_test, "y_test_final")


y_val = np.array([classes[text] for text in loadData("y_val")])
saveData(y_val, "y_val_final")