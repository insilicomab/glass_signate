# -*- coding: utf-8 -*-

# ライブラリの読み込み
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# データの読み込み
train = pd.read_csv('data/train.tsv', sep='\t')

# 学習用データの確認
print(train.head())

# 'id'の削除と確認
train = train.drop(train.columns[[0]], axis=1)
print(train.head())
print(train.shape)

# pandas-profiling
profile = ProfileReport(train)
profile.to_file('pandas_profiling/profile_report.html')

'''
予測したガラスの種類
1=building_windows_float_processed
2=building_windows_non_float_processed
3=vehicle_windows_float_processed
4=vehicle_windows_non_float_processed
5=containers
6=tableware
7=headlamps
'''