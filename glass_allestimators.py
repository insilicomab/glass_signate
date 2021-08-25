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
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# 3分割する
folds = 3
skf = StratifiedKFold(n_splits=folds)

# モデリングと正解率
model_list = []
allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms:
    
    if name == 'ClassifierChain':
        continue
    
    elif name == 'MultiOutputClassifier':
        continue
    
    elif name == 'OneVsOneClassifier':
        continue
    
    elif name == 'OneVsRestClassifier':
        continue
    
    elif name == 'OutputCodeClassifier':
        continue
    
    elif name == 'StackingClassifier':
        continue
    
    elif name == 'VotingClassifier':
        continue
    
    else:
        clf = algorithm()
        try:  # Errorがでるものがあるので、try文を入れる
            if hasattr(clf,"score"):
            # クロスバリデーション
                scores = cross_val_score(clf, X_train, Y_train, cv=skf)
                model_list.append(f"{name:<35}の正解率= {np.mean(scores)}")
    
        except:
            pass

# 表示
kekka = '\n'.join(model_list) # 文字列を改行させる
print(kekka)

"""
予測精度：
AdaBoostClassifier                 の正解率= 0.6341991341991342
BaggingClassifier                  の正解率= 0.7480519480519481
BernoulliNB                        の正解率= 0.5601731601731601
CalibratedClassifierCV             の正解率= 0.5974025974025974
CategoricalNB                      の正解率= nan
ComplementNB                       の正解率= nan
DecisionTreeClassifier             の正解率= 0.6917748917748918
DummyClassifier                    の正解率= 0.43982683982683984
ExtraTreeClassifier                の正解率= 0.6636363636363637
ExtraTreesClassifier               の正解率= 0.774891774891775
GaussianNB                         の正解率= 0.5883116883116883
GaussianProcessClassifier          の正解率= 0.6995670995670996
GradientBoostingClassifier         の正解率= 0.6718614718614718
HistGradientBoostingClassifier     の正解率= 0.7004329004329004
KNeighborsClassifier               の正解率= 0.7190476190476189
LabelPropagation                   の正解率= 0.7658008658008658
LabelSpreading                     の正解率= 0.7562770562770563
LinearDiscriminantAnalysis         の正解率= 0.6337662337662338
LinearSVC                          の正解率= 0.6337662337662338
LogisticRegression                 の正解率= 0.6432900432900432
LogisticRegressionCV               の正解率= 0.6441558441558441
MLPClassifier                      の正解率= 0.7086580086580087
MultinomialNB                      の正解率= nan
NearestCentroid                    の正解率= 0.5134199134199134
NuSVC                              の正解率= nan
PassiveAggressiveClassifier        の正解率= 0.522077922077922
Perceptron                         の正解率= 0.5956709956709957
QuadraticDiscriminantAnalysis      の正解率= 0.5606060606060606
RadiusNeighborsClassifier          の正解率= nan
RandomForestClassifier             の正解率= 0.7190476190476189
RidgeClassifier                    の正解率= 0.5874458874458874
RidgeClassifierCV                  の正解率= 0.5783549783549784
SGDClassifier                      の正解率= 0.5406926406926408
SVC                                の正解率= 0.6718614718614719
"""