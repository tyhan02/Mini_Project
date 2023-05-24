#!/usr/bin/env python
# coding: utf-8

# In[6]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[9]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from catboost import CatBoostRegressor

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

x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)

x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)
x_encoded = x_encoded.replace([np.inf, -np.inf], np.nan)
mean_values = x_encoded.mean()
x_encoded = x_encoded.fillna(mean_values)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, train_size=0.7, random_state=80, shuffle=True
)

feature_name = x.columns.tolist()
print(feature_name)

# Rest of the code for feature selection

from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.fit(x_train, y_train)

thresholds = np.squeeze(model.feature_importances_)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selecttion_model = CatBoostRegressor()
    selecttion_model.fit(select_x_train, y_train)
    y_predict = selecttion_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, acc:%.2f%%" % (thresh, select_x_train.shape[1], score * 100))

    # 컬럼명 출력
    selected_feature_indices = selection.get_support(indices=True)
    selected_feature_names = [feature_name[i] for i in selected_feature_indices]
    print(selected_feature_names)


# In[ ]:




