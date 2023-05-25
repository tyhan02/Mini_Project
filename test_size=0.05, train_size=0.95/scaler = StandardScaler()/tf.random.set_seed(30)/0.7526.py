#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[4]:


# 1. x, y Data
x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff','10~11_takeoff']].copy() #11~12 뺌



x['takeon_avg_6~8'] = (x['6~7_ride'] + x['7~8_ride']) / 2
x['takeon_avg_8~10'] = (x['8~9_ride'] + x['9~10_ride']) / 2
x['takeon_avg_10~12'] = (x['10~11_ride'] + x['11~12_ride']) / 2
x['takeon_avg_ride'] = (x['takeon_avg_6~8'] + x['takeon_avg_8~10'] + x['takeon_avg_10~12']) / 3







x['date'] = pd.to_datetime(datasets['date']) 
y = datasets[['18~20_ride']]

x['date'] = pd.to_datetime(x['date'])
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday
x = x.drop('date', axis=1) 


# In[5]:


x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)
x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)

# x_encoded = x_encoded.replace([np.inf, -np.inf], np.nan)
# mean_values = x_encoded.mean()
# x_encoded = x_encoded.fillna(mean_values)


# In[6]:


###########################################기본 전처리


# In[7]:


import tensorflow as tf
tf.random.set_seed(30) # weight 난수값 조정


# In[8]:



# print(x.info())
# print(x)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.05, train_size=0.95, random_state=80, shuffle=True
) # 7:3에서 바꿈
print(x)


# In[10]:


#kfold
n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=200)


# In[11]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

param = {
    'n_estimators': [3947],
    'depth': [16],
    'fold_permutation_block': [237],
    'learning_rate': [0.8989964556692867],
    'od_pval': [0.6429734179569129],
    'l2_leaf_reg': [2.169943087966259],
    'random_state': [1417]
}

bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    max_features=7,
    n_estimators=100,
    n_jobs=-1,
    random_state=62
)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)

depth = param['depth'][0]
l2_leaf_reg = param['l2_leaf_reg'][0]
border_count = param['fold_permutation_block'][0]

print(f"depth: {depth}")
print(f"l2_leaf_reg: {l2_leaf_reg}")
print(f"border_count: {border_count}")


# In[12]:


# # MinMaxScaler 0.711 0.709
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# StandardScaler 0.715 0.7151
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# RobustScaler 0.709 0.709
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# MaxAbsScaler 0.718 0.710 0.710
# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# In[13]:


# # PowerTransformer 0.718 0.717 0.709 0.703
# scaler = PowerTransformer()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # QuantileTransformer 0.714 0.711
# scaler = QuantileTransformer(output_distribution='normal')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # FunctionTransformer 0.718 0.702 0.707
# scaler = FunctionTransformer(np.log1p)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# In[14]:


# 2. 모델구성
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# In[15]:


# import optuna
# from optuna import Trial, visualization
# from optuna.samplers import TPESampler
# from sklearn.metrics import mean_absolute_error
# from catboost import CatBoostRegressor
# import matplotlib.pyplot as plt

# def objectiveCAT(trial: Trial, x_train, y_train, x_test):
#     param = {
#         'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
#         'depth' : trial.suggest_int('depth', 8, 16),
#         'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
#         'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
#         'od_pval' : trial.suggest_float('od_pval', 0, 1),
#         'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
#         'random_state' :trial.suggest_int('random_state', 1, 2000)
#     }
#     # 학습 모델 생성
#     model = CatBoostRegressor(**param)
#     CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
#     # 모델 성능 확인
#     score = r2_score(CAT_model.predict(x_test), y_test)
#     return score

# # MAE가 최소가 되는 방향으로 학습을 진행
# # TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
# study = optuna.create_study(direction='maximize', sampler=TPESampler())

# # n_trials 지정해주지 않으면, 무한 반복
# study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5)
# print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, 
#                                                   study.best_trial.params))

# # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
# print(optuna.visualization.plot_param_importances(study))
# # 하이퍼파라미터 최적화 과정을 확인
# optuna.visualization.plot_optimization_history(study)
# plt.show()


# In[16]:


#3. 훈련 및 평가예측
xgb = XGBRegressor()
cat = CatBoostRegressor()
lgbm = LGBMRegressor()

regressors = [cat, xgb, lgbm]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4}'.format(class_names, score))


# In[17]:


# MSE 계산
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_predict)

# MSE 값 출력
print("MSE:", mse)


# In[18]:


#4. 시각화 - 산점도 그래프 그리기
plt.figure(figsize=(8, 6))
plt.scatter(y_test.values.ravel(), y_predict)
plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()


# In[19]:


#4. 시각화 - 예측 오차의 분포 그래프 그리기
error = y_predict - y_test.values.ravel()
plt.figure(figsize=(8, 6))
plt.hist(error, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Prediction Error Distribution')
plt.show()


# In[20]:


#4. 시각화 - 상관계수 히트맵
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(20, 15)})
sns.heatmap(data=datasets.corr(),
           square = True,
            annot = True,
            cbar = True,
            cmap = 'coolwarm'
           )
plt.show()


# In[ ]:




