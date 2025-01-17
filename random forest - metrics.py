import psycopg2 as postgres
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

path = "./Temperature_&_GHI_ 2017_2020_myLocality_cleaned.csv"
df = pd.read_csv(path)
df

# Data Cleaning by replacing 0 with NaN and counting non-zero integer to have a correct weightage for GHI & Temp Values for prediction

df = df.fillna(0)
nonzero_mean = df[df!=0].mean()
nonzero_mean

cols = [0,1,2,3,4]
X_input = df[df.columns[cols]].values

Y_temp = df[df.columns[5]].values

Y_GHI = df[df.columns[6]].values

Y_temp, Y_GHI

X_train, X_test, X_temp_train, X_temp_test = train_test_split(X_input, Y_temp, random_state = 32)
Y_train, Y_test, Y_GHI_train, Y_GHI_test = train_test_split(X_input, Y_GHI, random_state = 32)

Rfr1 = RandomForestRegressor()
Rfr2 = RandomForestRegressor()

Rfr1.fit(X_train, X_temp_train)
Rfr2.fit(Y_train, Y_GHI_train)

X_predicted_test = Rfr1.predict(X_test)
Y_predicted_GHI = Rfr2.predict(Y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(X_temp_test, X_predicted_test)

mse = mean_squared_error(X_temp_test, X_predicted_test)

rmse = np.sqrt(mse)

r2 = r2_score(X_temp_test, X_predicted_test)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10, 6))

sns.kdeplot(X_temp_test, label='Actual Temperature', color='green', fill=True)
sns.kdeplot(X_predicted_test, label='Predicted Temperature', color='red', fill=True)
plt.figure(figsize=(10, 6))

sns.kdeplot(Y_GHI_test, label='Actual GHI', color='blue', fill=True)
sns.kdeplot(Y_predicted_GHI, label='Predicted GHI', color='orange', fill=True)

plt.title('KDE: Actual vs Predicted GHI')
plt.xlabel('GHI')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.title('KDE: Actual vs Predicted Temperature')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend()
plt.show()