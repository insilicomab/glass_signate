# -*- coding: utf-8 -*-

'''
データの精査
'''

# ライブラリのインポート
import pandas as pd
import numpy as np

# データの読み込み
train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')

# データの確認
print(train.head())
print(test.head())

# データの結合
df = pd.concat([train, test], sort=False) # concat()関数で縦（行）方向に結合

# 'id'の削除と確認
df = df.drop(train.columns[[0]], axis=1)
print(train.head())
print(train.shape)
print(len(train), len(test), len(df))

# 欠損値の確認
df.isnull().sum()

'''
特徴量エンジニアリング
'''

# ライブラリの読み込み
from sklearn.preprocessing import StandardScaler

# 説明変数の標準化
col=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'] # 標準化するカラムの指定
scaler = StandardScaler()
df[col] = scaler.fit_transform(df[col])
print(df.head())

# trainとtestに分配
train = df[:len(train)] # [:len(train)]は「インデクシング」と呼ばれ、[(開始位置) : (終了位置)]のように指定
test = df[len(train):]

# 説明変数と目的変数に分配
Y_train = train['Type']
X_train = train.drop(['Type', 'Ba', 'Fe'], axis=1)
X_test = test.drop(['Type', 'Ba', 'Fe'], axis=1)

'''
モデリング
'''

# ライブラリのインポート
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 学習用データと検証用データに分ける
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, 
                                                    test_size = 0.3,
                                                    random_state = 0,
                                                    stratify = Y_train) # 層化サンプリング

# Accuracyを格納するリスト
train_accuracy = []
test_accuracy = []

for k in range(1, 21):
    # kを渡してインスタンスを生成し、データをfitさせてモデルを作成
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    # scoreで正解率を取得し、順次格納
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_valid, y_valid))

# 正解率をnumpyの配列に変換
train_accuracy = np.array(train_accuracy)
test_accuracy = np.array(test_accuracy)

# 訓練用・テスト用の正解率の推移の描画
plt.figure(figsize=(6, 4))
plt.plot(range(1,21), train_accuracy, label='train')
plt.plot(range(1,21), test_accuracy, label='test')
plt.xticks(np.arange(1, 21, 1)) # x軸目盛
plt.xlabel('number of k')
plt.ylabel('accuracy')
plt.title('transition of accuracy')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
difference = np.abs(train_accuracy - test_accuracy) # 正解率の差分のを計算
plt.plot(range(1,21), difference, label='difference')
plt.xticks(np.arange(1, 21, 1)) # x軸目盛
plt.xlabel('number of k')
plt.ylabel('difference(train - test)')
plt.title('transition of difference(train - test)')
plt.grid()
plt.legend()
plt.show()

# 予測
k = 1 # kの個数を指定
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

'''
提出
'''

# 提出用サンプルの読み込みと確認
sub = pd.read_csv('./data/sample_submit.csv', header=None)
print(sub.head())

# 予測データの置き換え
sub[1] = y_pred.astype(int) # 整数型に変換したものを置き換える

# 提出用ファイルの出力
sub.to_csv('./submission/glass_knn.csv', header=None, index=None)
