#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
print(np.__version__)


# In[37]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import GridSearchCV
import time


# In[38]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[39]:


# 1. x, y Data
x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff', '10~11_takeoff']].copy()

x['date'] = pd.to_datetime(datasets['date']) 
y = datasets[['18~20_ride']]

x['date'] = pd.to_datetime(x['date'])
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday

x = x.drop('date', axis=1) 


# In[40]:


print(x.info())

print(x)


# In[41]:


x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)

x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)
x_encoded = x_encoded.replace([np.inf, -np.inf], np.nan)
mean_values = x_encoded.mean()
x_encoded = x_encoded.fillna(mean_values)


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, train_size=0.7, random_state=100, shuffle=True
)


# In[64]:


# 2. 모델구성
from catboost import CatBoostRegressor
model = CatBoostRegressor()


# In[65]:


# sclar
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[66]:


#kfold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)


# In[67]:


# param = {'max_features': [7], 'n_estimators': [100], 'random_state': [62]}

# # 모델 (Bagging)
# bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), max_features=7,
#                            n_estimators=100, n_jobs=-1, random_state=62)
# model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)


# In[68]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

param = {'max_features': [7], 'n_estimators': [100], 'random_state': [62]}
depth, l2_leaf_reg, border_count, bagging_temperature, max_ctr_complexity = 10, 3, 5, 0.5, 4

bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), max_features=7,
                 n_estimators=100, n_jobs=-1, random_state=62)

model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)

print(f"depth: {depth}")
print(f"l2_leaf_reg: {l2_leaf_reg}")
print(f"border_count: {border_count}")
print(f"bagging_temperature: {bagging_temperature}")
print(f"max_ctr_complexity: {max_ctr_complexity}")


# In[69]:


# 3. 훈련
start_time = time.time()
hist=model.fit(x_train, y_train.values.ravel())
end_time = time.time() - start_time

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
print('R2 Score:', r2)
print('MSE:', mse)


# In[70]:


# 4. 평가 예측
#loss = model.score(x_test, y_test.values.ravel())
#print('loss:', loss)

#y_predict = model.predict(x_test)


# R2 score
#mse = mean_squared_error(y_test, y_predict)
#r2 = r2_score(y_test, y_predict)
#print('r2 score:', r2)


# In[71]:


result = model.score(x_test, y_predict)
print('최적의 파라미터 : ', model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')
print('Bagging 결과 : ', result)


# In[72]:


model.fit(x_train, y_train)


# In[73]:


y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
print(score)


# In[74]:


#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

#4. 평가, 예측
xgb = XGBRegressor()
cat = CatBoostRegressor()
lgbm = LGBMRegressor()

regressors = [cat, xgb, lgbm]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))


# In[75]:


# 산점도 그래프 그리기
plt.figure(figsize=(30, 20))
plt.scatter(y_test.values.ravel(), y_predict)
plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()


# In[76]:


# 예측 오차의 분포 그래프 그리기
error = y_predict - y_test.values.ravel()
plt.figure(figsize=(8, 6))
plt.hist(error, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Prediction Error Distribution')
plt.show()


# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[78]:


sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(30, 25)})
sns.heatmap(data=datasets.corr(),
           square = True,
            annot = True,
            cbar = True,
            cmap = 'coolwarm'
           )
plt.show()


# In[ ]:




