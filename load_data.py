from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import mlflow
import tempfile

def load_data():
    """
    based on https://github.com/dbczumar/mlflow-keras-mnist/blob/master/train.py
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows,img_cols = x_train.shape[1],x_train.shape[2]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    num_classes = len(np.unique(y_train))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

print("before main")
if __name__ == "__main__":
    print("after main")
    with mlflow.start_run() as mlrun:
        print("after run start")
        dir_path = tempfile.mkdtemp()
        (x_train, y_train), (x_test, y_test) = load_data()
        paths = {"x_train" : os.path.join(dir_path,"x_train.npy"),
        "y_train": os.path.join(dir_path,"y_train.npy"),
        "x_test": os.path.join(dir_path,"x_test.npy"),
        "y_test": os.path.join(dir_path,"y_test.npy")}
        np.save(paths["x_train"],x_train)
        np.save(paths["y_train"],y_train)
        np.save(paths["x_test"],x_test)
        np.save(paths["y_test"],y_test)
        print("before logs")
        mlflow.log_artifacts(dir_path)