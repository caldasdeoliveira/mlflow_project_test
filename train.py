import pandas as pd
import numpy as np
import sys
import argparse


from tensorflow import keras
from tensorflow.keras import layers

import mlflow
import mlflow.keras

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import load_data

def eval_metrics(actual, pred):
  f1 = f1_score(actual, pred)
  acc = accuracy_score(actual, pred)
  roc_auc = roc_auc_score(actual, pred)
  return f1, acc, roc_auc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size','-b', type=int)
    parser.add_argument('--epochs','-e', type=int)
    parser.add_argument('--alpha','-a', type=float)
    parser.add_argument('--dropout','-d', type=float)

    data = sys.argv[0]
    batch_size = args.batch_size
    epochs = args.epochs
    alpha = args.alpha
    dropout = args.dropout
    
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape = x_train.shape[1:]

    with mlflow.start_run():

        mlflow.keras.autolog()
        model = keras.Sequential([
            keras.Input(shape = input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(9, activation="softmax"),
        ])


        optimizer = keras.optimizers.Adam(learning_rate=alpha)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[accuracy_score, f1_score, roc_auc_score])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)