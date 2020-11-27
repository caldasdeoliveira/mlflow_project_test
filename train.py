import pandas as pd
import numpy as np
import argparse
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import load_data_folder

def eval_metrics(actual, pred):
  metrics = {}
  metrics["f1"] = f1_score(actual, pred)
  metrics["acc"] = accuracy_score(actual, pred)
  metrics["roc_auc"] = roc_auc_score(actual, pred)
  return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-dt', type=str)
    parser.add_argument('--C','-c', type=float)
    parser.add_argument('--kernel','-k', type=str)
    parser.add_argument('--gamma','-g', type=str)
    parser.add_argument('--val_split','-v', type=float)

    args = parser.parse_args()

    data = args.data
    val_split = args.val_split
    C = args.C 
    gamma = args.gamma
    kernel = args.kernel

    params = { "val_split": val_split, "C": C, "gamma": gamma, "kernel": kernel}
    
    (x_train, y_train), (x_test, y_test) = load_data_folder(data)

    with mlflow.start_run():

        mlflow.log_params(params)
        mlflow.sklearn.autolog()

        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42)
        
        model = SVC(C=C, kernel=kernel, gamma=gamma)

        results = model.fit( X_train, Y_train)

        metrics = eval_metrics( Y_val, model.predict(X_val))

        mlflow.log_metrics(metrics)

