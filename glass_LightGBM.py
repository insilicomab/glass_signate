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
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean

# 3分割する
folds = 3
skf = StratifiedKFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    # 多値分類問題
    'objective': 'multiclass',
    # クラス数は7
    'num_class': 7
}

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
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=100, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする
    score = accuracy_score(y_valid, y_pred_max)
    print(score)
    
    models.append(model)
    scores.append(score)
    oof[val_index] = y_pred_max
    
    # 混同行列の作成
    cm = confusion_matrix(y_valid, y_pred_max)
    
    # heatmapによる混同行列の可視化
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()
    
    
# 平均accuracy scoreを計算する
print(mean(scores))

# 現状の予測値と実際の値の違いを可視化
actual_pred_df = pd.DataFrame({
    'actual':Y_train,
    'pred': oof})

actual_pred_df.plot(figsize=(12,5))

# 特徴量重要度の表示
for model in models:
    lgb.plot_importance(model, importance_type='gain',
                        max_num_features=15)

"""
予測精度：
0.7566137566137566
"""

# ライブラリのインポート
from scipy import stats

# 説明変数と目的変数を指定
X_test = test.drop(['Unnamed: 0', 'Ba', 'Fe'], axis=1)

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    pred_max = np.argmax(pred, axis=1)
    preds.append(pred_max)
    
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

# ダミー変数をもとに戻す
sub[1].replace([0,1,2,3,4,5,6], [1,2,3,4,5,6,7], inplace=True)

# ファイルのエクスポート
sub.to_csv('./submission/glass_LightGBM.csv', header=None, index=None)

"""
スコア：

"""