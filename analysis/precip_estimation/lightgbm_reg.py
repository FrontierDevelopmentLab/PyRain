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
val_path = "./sim_samples_unb_val.json"

with open(train_path, "r") as f:
    train = json.load(f)
    y_train = np.concatenate([np.array(t[1]) for t in train]) # use 2 for classification
    X_train = np.stack([t[0] for t in train])

with open(test_path, "r") as f:
    test = json.load(f)
    y_test = np.concatenate([np.array(t[1]) for t in test])
    X_test = np.stack([t[0] for t in test])

with open(val_path, "r") as f:
    val = json.load(f)
    y_val = np.concatenate([np.array(t[1]) for t in val])
    X_val = np.stack([t[0] for t in val])
    y_val_lst = [[],[],[],[]]
    X_val_lst = [[],[],[],[]]
    for c in range(4):
        y_val_lst[c] = np.concatenate([np.array(t[1]) for t in val if t[2][0]==c])
        X_val_lst[c] = np.stack(t[0] for t in val if t[2][0]==c)

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['rmse'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}

# train
print("Setting up regressor...")
gbm = lgb.LGBMRegressor(**hyper_params)

print("Setting up fit... {},{} -- {},{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='RMSE',
        early_stopping_rounds=1000)

undo_tp_train = lambda x: (np.exp(x) - 1) * nl_train["imerg5625/precipitationcal"]["std"]
undo_tp_test = lambda x: (np.exp(x) - 1) * nl_train["imerg5625/precipitationcal"]["std"]
undo_tp_val = lambda x: (np.exp(x) - 1) * nl_train["imerg5625/precipitationcal"]["std"]

y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
rmse_train = mean_squared_error(undo_tp_train(y_pred), undo_tp_train(y_train)) ** 0.5
rmse_train_log = mean_squared_error(y_pred, y_train) ** 0.5
print('The rmse of train is:', rmse_train)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
rmse_test = mean_squared_error(undo_tp_test(y_pred), undo_tp_test(y_test)) ** 0.5
rmse_test_log = mean_squared_error(y_pred, y_test) ** 0.5
print('The test of test is:', rmse_test)

y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration_)
rmse_val = mean_squared_error(undo_tp_val(y_pred), undo_tp_val(y_val)) ** 0.5
rmse_val_log = mean_squared_error(y_pred, y_val) ** 0.5
print('The test of val is:', rmse_val)

rmse_valc_lst = []
rmse_valc_log_lst = []
for c in range(4):
    y_pred = gbm.predict(X_val_lst[c], num_iteration=gbm.best_iteration_)
    rmse_valc = mean_squared_error(undo_tp_val(y_pred), undo_tp_val(y_val_lst[c])) ** 0.5
    rmse_valc_log = mean_squared_error(y_pred, y_val_lst[c]) ** 0.5
    print('The test of val-{} is:'.format(c), rmse_valc)
    rmse_valc_lst.append(rmse_valc)
    rmse_valc_log_lst.append(rmse_valc_log)

# Finished
print("Finished!")
res = {"rmse_train": rmse_train,
       "rmse_test": rmse_test,
       "rmse_val": rmse_val,
       "rmse_train_log": rmse_train_log,
       "rmse_test_log": rmse_test_log,
       "rmse_val_log": rmse_val_log,
       "rmse_valc": rmse_valc_lst,
       "rmse_valc_log": rmse_valc_log_lst}


print("RES: ", res)

with open("{}.json".format(tag), "w") as f:
    json.dump(res, f)

gbm.save_model('{}.txt'.format(tag), num_iteration=model.best_iteration)

