import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI

app = FastAPI()




## Dataset

dataset = pd.read_csv('./datasets/ufc-fighters-statistics.csv')

dataset = dataset.drop(columns=['name','nickname','reach_in_cm', 'date_of_birth'])

dataset['total_fights'] = dataset['wins'] + dataset['losses'] + dataset['draws']

dataset['lose_average'] = dataset['losses'] * 100 / dataset['total_fights']
dataset['win_average'] = dataset['wins'] * 100 / dataset['total_fights']
dataset['draw_average'] = dataset['draws'] * 100 / dataset['total_fights']


# Remove fighters with very few fights
dataset = dataset[dataset['total_fights'] >= 3]


stance_dummies = pd.get_dummies(dataset['stance'], prefix='Stance', dtype='float')

X = pd.concat([dataset, stance_dummies], axis=1)
dataset = dataset.dropna()


y = dataset['lose_average']
X = dataset.drop(columns=['losses','wins','draws','win_average','lose_average','draw_average','total_fights','stance'])


print(dataset.columns.values)

# Import Model
from catboost import CatBoostRegressor
# make a regressor object variable of the regressor class RandomForestRegressor
# n_estimators is how many regression trees, random state 0 for consistency of model on data
regressor = CatBoostRegressor(iterations=100, verbose=False, loss_function='RMSE')
# fitting the data
regressor.fit(X, y)

y_pred = regressor.predict(X)
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.reshape(y,(len(y),1))),1))

joblib.dump(regressor, 'ufc_win_predictor.pkl')
joblib.dump(X.columns.tolist(), 'ufc_features.pkl')