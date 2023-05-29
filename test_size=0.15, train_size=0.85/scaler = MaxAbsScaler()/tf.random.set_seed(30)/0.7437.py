#!/usr/bin/env python
# coding: utf-8

# In[27]:


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
from sklearn.ensemble import RandomForestRegressor


# In[28]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[29]:


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
y = datasets[['18~20_ride']]

x['date'] = pd.to_datetime(x['date'])
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday
x = x.drop('date', axis=1) 

# 최대값 중앙값 등
# Calculate statistical aggregations
ride_columns = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
takeoff_columns = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff']

# Calculate mean, median, standard deviation
# x['ride_mean'] = x[ride_columns].mean(axis=1)
# x['ride_median'] = x[ride_columns].median(axis=1)
# x['ride_std'] = x[ride_columns].std(axis=1)

# x['takeoff_mean'] = x[takeoff_columns].mean(axis=1)
x['takeoff_median'] = x[takeoff_columns].median(axis=1)
# x['takeoff_std'] = x[takeoff_columns].std(axis=1)

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


# In[30]:


# test.csv
test = pd.read_csv(path + 'test.csv')
test_x = test[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff','10~11_takeoff']].copy() #11~12 뺌

# ride 카테고리
test_x['takeon_avg_6~8'] = (test_x['6~7_ride'] + test_x['7~8_ride']) / 2
test_x['takeon_avg_8~10'] = (test_x['8~9_ride'] + test_x['9~10_ride']) / 2
test_x['takeon_avg_10~12'] = (test_x['10~11_ride'] + test_x['11~12_ride']) / 2
test_x['takeon_avg_ride'] = (test_x['takeon_avg_6~8'] + test_x['takeon_avg_8~10'] + test_x['takeon_avg_10~12']) / 3

# takeoff 카테고리
test_x['takeoff_avg_6~8'] = (test_x['6~7_takeoff'] + test_x['7~8_takeoff']) / 2
test_x['takeoff_avg_8~11'] = (test_x['8~9_takeoff'] + test_x['9~10_takeoff']+ test_x['10~11_takeoff']) / 3
test_x['takeon_avg_takeoff'] = (test_x['takeoff_avg_6~8'] + test_x['takeoff_avg_8~11'] ) / 2

test_x['date'] = pd.to_datetime(datasets['date']) 

test_x['date'] = pd.to_datetime(test_x['date'])
test_x['year'] = test_x['date'].dt.year
test_x['month'] = test_x['date'].dt.month
test_x['day'] = test_x['date'].dt.day
test_x['weekday'] = test_x['date'].dt.weekday
test_x = test_x.drop('date', axis=1) 

#
test_x['is_weekend'] = np.where(test_x['weekday'] < 5, 0, 1)
test_x['in_out'] = test_x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(test_x['station_name'].unique())}
test_x['station_name'] = test_x['station_name'].map(station_name_mapping)

test_x_encoded = pd.get_dummies(test_x, columns=['station_name'])
test_x_encoded = test_x_encoded.fillna(0)

test_x = pd.get_dummies(test_x, columns=['station_name'])
test_x = test_x_encoded.fillna(0)

# x_encoded = x_encoded.replace([np.inf, -np.inf], np.nan)
# mean_values = x_encoded.mean()
# x_encoded = x_encoded.fillna(mean_values)


# In[31]:


# x_encoded = x_encoded.replace([np.inf, -np.inf], np.nan)
# mean_values = x_encoded.mean()
# x_encoded = x_encoded.fillna(mean_values)


# In[32]:


###########################################기본 전처리


# In[33]:


import tensorflow as tf
tf.random.set_seed(30) # weight 난수값 조정


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, train_size=0.85, random_state=80, shuffle=True
) # 7:3에서 바꿈
print(x)


# In[ ]:


# bagging, optuna 의 최적의 param 적용하여 최적의 조합을 서치
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

param = {
    'n_estimators': [559], 
              'depth': [9], 
              'fold_permutation_block': [120], 
             'learning_rate': [0.8767296308401672], 
              'od_pval': [0.8042245591717342], 
             'l2_leaf_reg': [1.9840469251833572], 
             'random_state': [1510]
}

bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    max_features=7,
    n_estimators=100,
    n_jobs=-1,
    random_state=62
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)


# In[ ]:


# MaxAbsScaler 0.7337   0.7337   MSE: 6.94459172588048
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


# 2. 모델구성
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# In[ ]:


model = CatBoostRegressor()

# model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
print('catboost',score)


# In[ ]:


498:	learn: 2.1676507	total: 6.88s	remaining: 6.91s
499:	learn: 2.1674314	total: 6.9s	remaining: 6.9s
500:	learn: 2.1667174	total: 6.91s	remaining: 6.88s
501:	learn: 2.1661994	total: 6.93s	remaining: 6.87s
502:	learn: 2.1655184	total: 6.94s	remaining: 6.86s
503:	learn: 2.1652117	total: 6.95s	remaining: 6.84s
504:	learn: 2.1645881	total: 6.97s	remaining: 6.83s
505:	learn: 2.1640020	total: 6.98s	remaining: 6.81s
506:	learn: 2.1636179	total: 6.99s	remaining: 6.8s
507:	learn: 2.1631768	total: 7.01s	remaining: 6.79s
508:	learn: 2.1624656	total: 7.02s	remaining: 6.77s
509:	learn: 2.1621282	total: 7.03s	remaining: 6.76s
510:	learn: 2.1612021	total: 7.04s	remaining: 6.74s
511:	learn: 2.1604888	total: 7.06s	remaining: 6.73s
512:	learn: 2.1600504	total: 7.07s	remaining: 6.71s
513:	learn: 2.1594461	total: 7.09s	remaining: 6.7s
514:	learn: 2.1591949	total: 7.1s	remaining: 6.69s
515:	learn: 2.1588557	total: 7.11s	remaining: 6.67s
516:	learn: 2.1579087	total: 7.13s	remaining: 6.66s
517:	learn: 2.1568760	total: 7.14s	remaining: 6.64s
518:	learn: 2.1567087	total: 7.15s	remaining: 6.63s
519:	learn: 2.1561589	total: 7.17s	remaining: 6.62s
520:	learn: 2.1556941	total: 7.18s	remaining: 6.6s
521:	learn: 2.1553012	total: 7.19s	remaining: 6.59s
522:	learn: 2.1543415	total: 7.21s	remaining: 6.57s
523:	learn: 2.1535273	total: 7.22s	remaining: 6.56s
524:	learn: 2.1529106	total: 7.24s	remaining: 6.55s
525:	learn: 2.1523511	total: 7.25s	remaining: 6.53s
526:	learn: 2.1520814	total: 7.26s	remaining: 6.52s
527:	learn: 2.1516025	total: 7.28s	remaining: 6.5s
528:	learn: 2.1499511	total: 7.29s	remaining: 6.49s
529:	learn: 2.1491314	total: 7.3s	remaining: 6.48s
530:	learn: 2.1485291	total: 7.32s	remaining: 6.46s
531:	learn: 2.1476653	total: 7.33s	remaining: 6.45s
532:	learn: 2.1472262	total: 7.35s	remaining: 6.44s
533:	learn: 2.1457783	total: 7.36s	remaining: 6.42s
534:	learn: 2.1451503	total: 7.37s	remaining: 6.41s
535:	learn: 2.1446021	total: 7.38s	remaining: 6.39s
536:	learn: 2.1439724	total: 7.4s	remaining: 6.38s
537:	learn: 2.1432073	total: 7.41s	remaining: 6.37s
538:	learn: 2.1427871	total: 7.42s	remaining: 6.35s
539:	learn: 2.1423202	total: 7.44s	remaining: 6.34s
540:	learn: 2.1420424	total: 7.45s	remaining: 6.32s
541:	learn: 2.1417302	total: 7.46s	remaining: 6.31s
542:	learn: 2.1413186	total: 7.48s	remaining: 6.29s
543:	learn: 2.1407758	total: 7.49s	remaining: 6.28s
544:	learn: 2.1402947	total: 7.5s	remaining: 6.26s
545:	learn: 2.1398552	total: 7.52s	remaining: 6.25s
546:	learn: 2.1394521	total: 7.53s	remaining: 6.24s
547:	learn: 2.1390364	total: 7.54s	remaining: 6.22s
548:	learn: 2.1385930	total: 7.56s	remaining: 6.21s
549:	learn: 2.1379130	total: 7.57s	remaining: 6.19s
550:	learn: 2.1373735	total: 7.58s	remaining: 6.18s
551:	learn: 2.1370599	total: 7.6s	remaining: 6.17s
552:	learn: 2.1363575	total: 7.61s	remaining: 6.15s
553:	learn: 2.1358668	total: 7.63s	remaining: 6.14s
554:	learn: 2.1354840	total: 7.64s	remaining: 6.13s
555:	learn: 2.1351352	total: 7.65s	remaining: 6.11s
556:	learn: 2.1341629	total: 7.67s	remaining: 6.1s
557:	learn: 2.1336635	total: 7.68s	remaining: 6.08s
558:	learn: 2.1329420	total: 7.7s	remaining: 6.07s
559:	learn: 2.1320377	total: 7.71s	remaining: 6.06s
560:	learn: 2.1315364	total: 7.72s	remaining: 6.04s
561:	learn: 2.1311531	total: 7.74s	remaining: 6.03s
562:	learn: 2.1308861	total: 7.75s	remaining: 6.02s
563:	learn: 2.1304090	total: 7.77s	remaining: 6s
564:	learn: 2.1297924	total: 7.78s	remaining: 5.99s
565:	learn: 2.1295704	total: 7.8s	remaining: 5.98s
566:	learn: 2.1289594	total: 7.81s	remaining: 5.97s
567:	learn: 2.1282461	total: 7.83s	remaining: 5.95s
568:	learn: 2.1277321	total: 7.84s	remaining: 5.94s
569:	learn: 2.1272752	total: 7.85s	remaining: 5.92s
570:	learn: 2.1267931	total: 7.87s	remaining: 5.91s
571:	learn: 2.1259745	total: 7.88s	remaining: 5.9s
572:	learn: 2.1249903	total: 7.9s	remaining: 5.88s
573:	learn: 2.1242084	total: 7.91s	remaining: 5.87s
574:	learn: 2.1236719	total: 7.93s	remaining: 5.86s
575:	learn: 2.1233654	total: 7.94s	remaining: 5.85s
576:	learn: 2.1227013	total: 7.96s	remaining: 5.83s
577:	learn: 2.1221590	total: 7.97s	remaining: 5.82s
578:	learn: 2.1217314	total: 7.98s	remaining: 5.8s
579:	learn: 2.1210376	total: 8s	remaining: 5.79s
580:	learn: 2.1202823	total: 8.01s	remaining: 5.78s
581:	learn: 2.1193692	total: 8.03s	remaining: 5.76s
582:	learn: 2.1187398	total: 8.04s	remaining: 5.75s
583:	learn: 2.1180796	total: 8.05s	remaining: 5.74s
584:	learn: 2.1177007	total: 8.07s	remaining: 5.72s
585:	learn: 2.1172241	total: 8.08s	remaining: 5.71s
586:	learn: 2.1166023	total: 8.1s	remaining: 5.7s
587:	learn: 2.1160649	total: 8.11s	remaining: 5.68s
588:	learn: 2.1157265	total: 8.13s	remaining: 5.67s
589:	learn: 2.1153164	total: 8.14s	remaining: 5.66s
590:	learn: 2.1143310	total: 8.15s	remaining: 5.64s
591:	learn: 2.1139722	total: 8.17s	remaining: 5.63s
592:	learn: 2.1135322	total: 8.18s	remaining: 5.61s
593:	learn: 2.1130116	total: 8.19s	remaining: 5.6s
594:	learn: 2.1126545	total: 8.21s	remaining: 5.59s
595:	learn: 2.1122911	total: 8.22s	remaining: 5.57s
596:	learn: 2.1117754	total: 8.23s	remaining: 5.56s
597:	learn: 2.1115648	total: 8.25s	remaining: 5.54s
598:	learn: 2.1109083	total: 8.26s	remaining: 5.53s
599:	learn: 2.1104670	total: 8.28s	remaining: 5.52s
600:	learn: 2.1099266	total: 8.29s	remaining: 5.5s
601:	learn: 2.1096402	total: 8.3s	remaining: 5.49s
602:	learn: 2.1092430	total: 8.31s	remaining: 5.47s
603:	learn: 2.1087640	total: 8.33s	remaining: 5.46s
604:	learn: 2.1081590	total: 8.34s	remaining: 5.45s
605:	learn: 2.1078152	total: 8.35s	remaining: 5.43s
606:	learn: 2.1074032	total: 8.37s	remaining: 5.42s
607:	learn: 2.1069079	total: 8.38s	remaining: 5.4s
608:	learn: 2.1062003	total: 8.4s	remaining: 5.39s
609:	learn: 2.1055867	total: 8.41s	remaining: 5.38s
610:	learn: 2.1051980	total: 8.42s	remaining: 5.36s
611:	learn: 2.1046977	total: 8.44s	remaining: 5.35s
612:	learn: 2.1040454	total: 8.45s	remaining: 5.33s
613:	learn: 2.1033667	total: 8.46s	remaining: 5.32s
614:	learn: 2.1031149	total: 8.48s	remaining: 5.31s
615:	learn: 2.1027496	total: 8.49s	remaining: 5.29s
616:	learn: 2.1022235	total: 8.51s	remaining: 5.28s
617:	learn: 2.1018728	total: 8.52s	remaining: 5.27s
618:	learn: 2.1007619	total: 8.53s	remaining: 5.25s
619:	learn: 2.1002115	total: 8.55s	remaining: 5.24s
620:	learn: 2.0997399	total: 8.56s	remaining: 5.23s
621:	learn: 2.0990860	total: 8.58s	remaining: 5.21s
622:	learn: 2.0987381	total: 8.59s	remaining: 5.2s
623:	learn: 2.0980665	total: 8.6s	remaining: 5.18s
624:	learn: 2.0974573	total: 8.62s	remaining: 5.17s
625:	learn: 2.0971314	total: 8.63s	remaining: 5.16s
626:	learn: 2.0955933	total: 8.64s	remaining: 5.14s
627:	learn: 2.0946134	total: 8.66s	remaining: 5.13s
628:	learn: 2.0941152	total: 8.67s	remaining: 5.11s
629:	learn: 2.0936000	total: 8.69s	remaining: 5.1s
630:	learn: 2.0930948	total: 8.7s	remaining: 5.09s
631:	learn: 2.0925511	total: 8.71s	remaining: 5.07s
632:	learn: 2.0922154	total: 8.73s	remaining: 5.06s
633:	learn: 2.0917836	total: 8.74s	remaining: 5.04s
634:	learn: 2.0912195	total: 8.75s	remaining: 5.03s
635:	learn: 2.0905392	total: 8.77s	remaining: 5.02s
636:	learn: 2.0901852	total: 8.78s	remaining: 5s
637:	learn: 2.0897584	total: 8.79s	remaining: 4.99s
638:	learn: 2.0892911	total: 8.81s	remaining: 4.97s
639:	learn: 2.0889843	total: 8.82s	remaining: 4.96s
640:	learn: 2.0886857	total: 8.83s	remaining: 4.95s
641:	learn: 2.0882364	total: 8.84s	remaining: 4.93s
642:	learn: 2.0880268	total: 8.86s	remaining: 4.92s
643:	learn: 2.0875934	total: 8.87s	remaining: 4.9s
644:	learn: 2.0872867	total: 8.88s	remaining: 4.89s
645:	learn: 2.0865152	total: 8.9s	remaining: 4.88s
646:	learn: 2.0859985	total: 8.91s	remaining: 4.86s
from sklearn.linear_model import Lassofrom sklearn.linear_model import Lasso

score = r2_score(y_test, y_predict)
print('catboost',score)


# In[ ]:


# MSE 계산
mse = mean_squared_error(y_test, y_predict)

# MSE 값 출력
print("MSE:", mse)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_predict)
plt.plot(y_test, y_predict, color='Red')
plt.show()


# In[ ]:


# #4. 시각화 - 산점도 그래프 그리기
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test.values.ravel(), y_predict)
# plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], 'k--', lw=2)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted')
# plt.show()


# In[ ]:


# #4. 시각화 - 예측 오차의 분포 그래프 그리기
# error = y_predict - y_test.values.ravel()
# plt.figure(figsize=(8, 6))
# plt.hist(error, bins=30)
# plt.xlabel('Prediction Error')
# plt.ylabel('Count')
# plt.title('Prediction Error Distribution')
# plt.show()


# In[ ]:


# #4. 시각화 - 상관계수 히트맵
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale = 1.2)
# sns.set(rc = {'figure.figsize':(20, 15)})
# sns.heatmap(data=datasets.corr(),
#            square = True,
#             annot = True,
#             cbar = True,
#             cmap = 'coolwarm'
#            )
# plt.show()


# In[ ]:


y_result = model.predict(test_x)
test_x['predicted_18~20_ride'] = y_result
print(test_x.head())


# In[ ]:




