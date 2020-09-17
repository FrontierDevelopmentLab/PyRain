import numpy as np
from collections import Counter
import pandas as pd
import lightgbm as lgb
import json
from sklearn.datasets import load_breast_cancer,load_boston,load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score

##### DEFINE TAG
tag = "sim_bal"
######

with open("./normalise/5625__16-04-01_12:00to17-12-31_11:00.json") as f:
    nl_train = json.load(f)

import json

train_path = "./sim_samples_bal_train.json"
test_path = "./sim_samples_unb_test.json"
val_path = "./sim_samples_bal_val.json"


with open(val_path, "r") as f:
    val = json.load(f)
    y_val = np.concatenate([np.array(t[1]) for t in val])
    X_val = np.stack([t[0] for t in val])
    y_val_lst = [[],[],[],[]]
    X_val_lst = [[],[],[],[]]
    for c in range(4):
        y_val_lst[c] = np.concatenate([np.array(t[1]) for t in val if t[2][0]==c])
        X_val_lst[c] = np.stack(t[0] for t in val if t[2][0]==c)

print(X_val_lst[0].shape)

print("Open model...")
import joblib
mod = joblib.load("gbmsim_bal.pkl")
print("Loading done...")

conf_matrix = np.zeros((4,4))
for i in range(4):
    print("Predict class {}".format(i))
    ypred = mod.predict(X_val_lst[i], num_iteration=mod.best_iteration_)
    print("Done predicting...")
    for p, t in zip(ypred, y_val_lst[i]):
        if p < 2.5:
            c = 0
        elif p >= 2.5 and p < 10.0:
            c = 1
        elif p >= 10.0 and p < 50.0:
            c = 2
        elif p >= 50.0:
            c = 3
        conf_matrix[i, c] += 1.0 / float(len(ypred))

print("CONF MATRIX:")
print(conf_matrix)

print("All done...")

