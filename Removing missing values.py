#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[3]:


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


# In[4]:


x = x.drop('date', axis=1) 
# print(x.info())
# print(x)


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


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, train_size=0.7, random_state=80, shuffle=True
)
print(x)


# In[7]:


#kfold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)


# In[8]:


x = train[['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride',
              '6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']]

# 이상치 식별을 위한 IQR 계산
Q1 = x.quantile(0.25)
Q3 = x.quantile(0.75)
IQR = Q3 - Q1

# 이상치 식별
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (x < lower_bound) | (x > upper_bound)

# 이상치 확인 결과 출력
print(outliers)

# Box Plot으로 이상치 확인
plt.figure(figsize=(12, 6))
x.boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot of Outliers")
plt.show()


# In[ ]:


plt.figure(figsize=(12, 6))
train.boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot of Outliers")
plt.show()

