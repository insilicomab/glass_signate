# -*- coding: utf-8 -*-
"""
コメント
ver2: 特徴量から'Ba', 'Fe'を削除
"""

'''
データの精査
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')
submission = pd.read_csv('./data/sample_submit.csv', header=None)

# データの確認
print(train.head(), train.dtypes)
print(test.head())
print(submission.head())

# データの結合
df = pd.concat([train, test], sort=False) # concat()関数で縦（行）方向に結合

# 'id'の削除と確認
df = df.drop(train.columns[[0]], axis=1)
print(train.head())
print(train.shape)
print(len(train), len(test), len(df))

# 欠損値の確認
print(df.isnull().sum())

'''
特徴量エンジニアリング
'''

# ダミー変数化
train['Type'].replace([1,2,3,4,5,6,7], [0,1,2,3,4,5,6], inplace=True)

'''
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 説明変数と目的変数を指定
X_train = train.drop(['Type', 'Unnamed: 0', 'Ba', 'Fe'], axis=1)
Y_train = train['Type']

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=0.3,
                                                      random_state=1234,
                                                      stratify=Y_train)

# パラメーターの設定
params = {'n_estimators':[300,400,500],
          'max_depth':[5,6,7,8,9,10] ,
          'n_jobs':[-1],
          'max_features':[3,4,5]}

# グリッドサーチ
grid = GridSearchCV(rf(), params, cv=5)
grid.fit(x_train, y_train)

# 最適なパラメーター
best_param= grid.best_params_
print(best_param)

# 最適なモデル
best_model = grid.best_estimator_

# 検証データの予測
y_pred= best_model.predict(x_valid)

# 正解率
print(accuracy_score(y_valid,y_pred))

"""
best_param={'max_depth': 8, 'max_features': 3, 'n_estimators': 400, 'n_jobs': -1}

予測精度：
0.7575757575757576
"""