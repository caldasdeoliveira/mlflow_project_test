import os
import numpy as np

def load_data_folder(data_path):
    x_train = np.load(os.path.join(data_path,"x_train.npy"))
    y_train = np.load(os.path.join(data_path,"y_train.npy"))
    x_test = np.load(os.path.join(data_path,"x_test.npy"))
    y_test = np.load(os.path.join(data_path,"y_test.npy"))
    return (x_train, y_train), (x_test, y_test)