import pandas as pd
import numpy as np
import argparse
import os

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import load_data_folder

def eval_metrics(actual, pred):
  metrics = {}
  #metrics["f1"] = f1_score(actual, pred)
  metrics["acc"] = accuracy_score(actual, pred)
  metrics["roc_auc"] = roc_auc_score(actual, pred)
  return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-dt', type=str)
    parser.add_argument('--C','-c', type=float)
    parser.add_argument('--dual','-d', type=bool)
    parser.add_argument('--penalty','-p', type=str)
    parser.add_argument('--val_split','-v', type=float)

    args = parser.parse_args()

    data = args.data
    val_split = args.val_split
    C = args.C 
    dual = args.dual
    penalty = args.penalty

    params = { "val_split": val_split, "C": C, "dual": dual, "penalty": penalty}
    
    (x_train, y_train), (x_test, y_test) = load_data_folder(data)

    with mlflow.start_run(experiment_id=1):

        mlflow.log_params(params)
        mlflow.sklearn.autolog()

        x_train = x_train.reshape((x_train.shape[0], 28*28))
        y_train = y_train.argmax(1)

        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42)
        
        model = LinearSVC(C=C, dual=dual, penalty=penalty)

        results = model.fit( X_train, Y_train)

        metrics = eval_metrics( Y_val, model.predict(X_val))

        mlflow.log_metrics(metrics)

