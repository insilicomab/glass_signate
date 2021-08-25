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
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean

# 3分割する
folds = 3
skf = StratifiedKFold(n_splits=folds)

# 説明変数と目的変数を指定
X_train = train.drop(['Type', 'Unnamed: 0', 'Ba', 'Fe'], axis=1)
Y_train = train['Type']

# 各foldごとに作成したモデルごとの予測値を保存
models = []
scores = []
oof = np.zeros(len(X_train))

for train_index, val_index in skf.split(X_train, Y_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]

    model = rf(max_depth=8,
               max_features=3,
               n_estimators=400,
               n_jobs = -1,
               random_state=1234)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    
    score = accuracy_score(y_valid, y_pred)
    print(score)
    
    models.append(model)
    scores.append(score)
    oof[val_index] = y_pred
    
    # 混同行列の作成
    cm = confusion_matrix(y_valid, y_pred)
    
    # heatmapによる混同行列の可視化
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()
    
    
# 平均accuracy scoreを計算する
print(mean(scores))

"""
予測精度：
0.7748677248677249
"""

'''
テストデータの予測
'''

# ライブラリのインポート
from scipy import stats

# 説明変数と目的変数を指定
X_test = test.drop(['Unnamed: 0', 'Ba', 'Fe'], axis=1)

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)
    
# アンサンブル学習
preds_array = np.array(preds)
pred = stats.mode(preds_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv('./data/sample_submit.csv', sep=',', header=None)
print(sub.head())
    
# 目的変数カラムの置き換え
sub[1] = pred.astype(int) # 整数型に変換したものを置き換える

# ファイルのエクスポート
sub.to_csv('./submission/glass_RandamForest.csv', header=None, index=None)

"""
スコア：
0.6074766
"""