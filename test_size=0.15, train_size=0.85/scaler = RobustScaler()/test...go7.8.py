#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# In[2]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[3]:


# # 1. x, y Data
# x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
#               'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
#               '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
#               '8~9_takeoff', '9~10_takeoff','10~11_takeoff']].copy() #11~12 뺌

# # ride 카테고리
# x['takeon_avg_6~8'] = (x['6~7_ride'] + x['7~8_ride']) / 2
# x['takeon_avg_8~10'] = (x['8~9_ride'] + x['9~10_ride']) / 2
# x['takeon_avg_10~12'] = (x['10~11_ride'] + x['11~12_ride']) / 2
# x['takeon_avg_ride'] = (x['takeon_avg_6~8'] + x['takeon_avg_8~10'] + x['takeon_avg_10~12']) / 3

# # takeoff 카테고리
# x['takeoff_avg_6~8'] = (x['6~7_takeoff'] + x['7~8_takeoff']) / 2
# x['takeoff_avg_8~11'] = (x['8~9_takeoff'] + x['9~10_takeoff']+ x['10~11_takeoff']) / 3
# x['takeon_avg_takeoff'] = (x['takeoff_avg_6~8'] + x['takeoff_avg_8~11'] ) / 2

# # date 카테코리
# x['date'] = pd.to_datetime(datasets['date']) 
# y = datasets[['18~20_ride']]

# x['date'] = pd.to_datetime(x['date'])
# x['year'] = x['date'].dt.year
# x['month'] = x['date'].dt.month
# x['day'] = x['date'].dt.day
# x['weekday'] = x['date'].dt.weekday
# x = x.drop('date', axis=1) 

# # 최대값 중앙값 등
# # Calculate statistical aggregations
# ride_columns = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
# takeoff_columns = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff']

# x['takeoff_median'] = x[takeoff_columns].median(axis=1)

# # Calculate maximum and minimum values
# x['ride_max'] = x[ride_columns].max(axis=1)
# x['ride_min'] = x[ride_columns].min(axis=1)

# # Add weekday/weekend feature
# x['is_weekend'] = np.where(x['weekday'] < 5, 0, 1)

# x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
# station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
# x['station_name'] = x['station_name'].map(station_name_mapping)
# x_encoded = pd.get_dummies(x, columns=['station_name'])
# x_encoded = x_encoded.fillna(0)

# x = pd.get_dummies(x, columns=['station_name'])
# x = x_encoded.fillna(0)


# In[4]:


# 1. x, y Data
x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff','10~11_takeoff']].copy() #11~12 뺌

# ride 카테고리
x['takeon_avg_6~8'] = (x['6~7_ride'] + x['7~8_ride']) / 2
x['takeon_avg_8~10'] = (x['8~9_ride'] + x['9~10_ride']) / 2
x['takeon_avg_10~12'] = (x['10~11_ride'] + x['11~12_ride']) / 2
x['takeon_avg_ride'] = (x['takeon_avg_6~8'] + x['takeon_avg_8~10'] + x['takeon_avg_10~12']) / 3

# takeoff 카테고리
x['takeoff_avg_6~8'] = (x['6~7_takeoff'] + x['7~8_takeoff']) / 2
x['takeoff_avg_8~11'] = (x['8~9_takeoff'] + x['9~10_takeoff']+ x['10~11_takeoff']) / 3
x['takeon_avg_takeoff'] = (x['takeoff_avg_6~8'] + x['takeoff_avg_8~11'] ) / 2


# date 카테코리
x['date'] = pd.to_datetime(datasets['date']) 
y = datasets['18~20_ride']


# In[5]:


datasets['date']


# In[6]:


import datetime
x['date'] = pd.to_datetime(x['date'])
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday

# import datetime

# # # 특정 공휴일 설정
# holiday_date01 = '2019-09-12'
# holiday_date02 = '2019-09-13'

# # 공휴일 요일 확인
# holiday01 = datetime.datetime.strptime(holiday_date01, '%Y-%m-%d')
# holiday02 = datetime.datetime.strptime(holiday_date02, '%Y-%m-%d')
# holiday_weekday01 = holiday01.weekday()
# holiday_weekday02 = holiday02.weekday()

# # 평일인 경우 해당 공휴일을 주말로 처리하는 새로운 요일 설정
# if holiday_weekday01 < 5:  # 월요일(0)부터 금요일(4)까지를 평일로 가정
#     holiday_weekday01 = 5  # 토요일(5)로 변경

# if holiday_weekday02 < 5:
#     holiday_weekday02 = 5

# # # 주말 여부를 나타내는 새로운 열 추가
# x['is_weekend_holiday'] = np.where((x['date'].dt.date == holiday.date()), 1, 0)
x = x.drop('date', axis=1)


# In[7]:


# 최대값 중앙값 등
# Calculate statistical aggregations
ride_columns = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
takeoff_columns = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff']

x['takeoff_median'] = x[takeoff_columns].median(axis=1)

# Calculate maximum and minimum values
x['ride_max'] = x[ride_columns].max(axis=1)
x['ride_min'] = x[ride_columns].min(axis=1)

# Add weekday/weekend feature
x['is_weekend'] = np.where(x['weekday'] < 5, 0, 1) 

x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)
x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)

x = pd.get_dummies(x, columns=['station_name'])
x = x_encoded.fillna(0)


# ###########################################기본 전처리

# In[8]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, train_size=0.85, random_state=80, shuffle=True
) # 7:3에서 바꿈


# In[9]:


# Data preprocessing
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[10]:


from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import numpy as np
model1 = CatBoostRegressor()
model2 = RandomForestRegressor()


# In[11]:


# Data preprocessing
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[12]:


# Define ensemble model
ensemble = VotingRegressor([('catboost', model1), ('randomforest', model2)])


# In[13]:


# Train the ensemble model
# ensemble = VotingRegressor([('catboost', model1), ('randomforest', model2)])


# In[ ]:


ensemble.fit(x_train, y_train)


# In[ ]:


y_pred = ensemble.predict(x_test_scaled)

# Transform the predictions back to the original scale
y_pred = np.expm1(y_pred)


# In[ ]:


# Evaluate the model
score = r2_score(np.expm1(y_test), y_pred)
mse = mean_squared_error(np.expm1(y_test), y_pred)


# In[ ]:


print('정확도:', score)
print('MSE:', mse)


# In[ ]:


# from catboost import CatBoostRegressor

# # Create a CatBoost Regressor model with optimized parameters
# model = CatBoostRegressor()


# In[ ]:


# #3. 훈련
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)


# In[ ]:


# # Evaluate the model
# score = r2_score(y_test, y_predict)
# print('정확도', score)


# In[ ]:


# # MSE 계산
# mse = mean_squared_error(y_test, y_predict)

# # MSE 값 출력
# print("MSE:", mse)


# In[ ]:




