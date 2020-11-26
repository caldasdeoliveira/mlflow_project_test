import pandas as pd
import numpy as np
import argparse
import os


from tensorflow import keras
from tensorflow.keras import layers

import mlflow
import mlflow.keras

# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import load_data_folder

# def eval_metrics(actual, pred):
#  f1 = f1_score(actual, pred)
#  acc = accuracy_score(actual, pred)
#  roc_auc = roc_auc_score(actual, pred)
#  return f1, acc, roc_auc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-dt', type=str)
    parser.add_argument('--batch_size','-b', type=int)
    parser.add_argument('--epochs','-e', type=int)
    parser.add_argument('--alpha','-a', type=float)
    parser.add_argument('--dropout','-d', type=float)
    parser.add_argument('--val_split','-v', type=float)

    args = parser.parse_args()

    data = args.data
    batch_size = args.batch_size
    epochs = args.epochs
    val_split = args.val_split
    alpha = args.alpha
    dropout = args.dropout
    
    (x_train, y_train), (x_test, y_test) = load_data_folder(data)
    input_shape = x_train.shape[1:]

    with mlflow.start_run(experiment_id=1):

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
            layers.Dense(10, activation="softmax"),
        ])

        model.summary()

        opt =keras.optimizers.Adam(learning_rate = alpha)

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy", "AUC"])

        results = model.fit( x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split)