import psycopg2 as postgres
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import datetime
import time

conn = postgres.connect(database="solar_power_prediction", user="postgres", password="root")
cursor = conn.cursor()

path = "./Temperature_&_GHI_ 2017_2020_myLocality_cleaned.csv"
df = pd.read_csv(path)
df

# Data Cleaning by replacing 0 with NaN and counting non-zero integer to have a correct weightage for GHI & Temp Values for prediction

df = df.fillna(0)
nonzero_mean = df[df!=0].mean()

cols = [0,1,2,3,4]
X_input = df[df.columns[cols]].values

Y_temp = df[df.columns[5]].values
Y_GHI = df[df.columns[6]].values

X_train, X_test, X_temp_train, X_temp_test = train_test_split(X_input, Y_temp, random_state = 32)
Y_train, Y_test, Y_GHI_train, Y_GHI_test = train_test_split(X_input, Y_GHI, random_state = 32)

temp_model = LinearRegression()
ghi_model = LinearRegression()

temp_model.fit(X_train, X_temp_train)
ghi_model.fit(Y_train, Y_GHI_train)

X_predicted_test = temp_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Mean Absolute Error (MAE)
mae = mean_absolute_error(X_temp_test, X_predicted_test)

# Mean Squared Error (MSE)
mse = mean_squared_error(X_temp_test, X_predicted_test)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R²)
r2 = r2_score(X_temp_test, X_predicted_test)

# Print the metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

import matplotlib.pyplot as plt

import seaborn as sns

# Plotting KDE for Actual vs Predicted Temperature
plt.figure(figsize=(10, 6))

# KDE plot for actual temperature
sns.kdeplot(X_temp_test, label='Actual Temperature', color='green', fill=True)

# KDE plot for predicted temperature
sns.kdeplot(X_predicted_test, label='Predicted Temperature', color='red', fill=True)

# Adding titles and labels
plt.title('KDE: Actual vs Predicted Temperature')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.show()

'''
current_time = datetime.datetime.now()
final_time = current_time # make a copy

while True:
    final_time = final_time + datetime.timedelta(minutes=10)
    copy_time_after_15 = final_time 

    time_after_15 = final_time.strftime("%Y-%m-%d %H:%M")
    print(time_after_15)

    time_after_15_list = copy_time_after_15.strftime("%Y,%m,%d,%H,%M")
    time_after_15_list = time_after_15_list.split(',')
    
    # Convert the list of strings into list of numerical values, as certain models expect in numerics and doesnt do conversions like random forest did, when a string list is passed.
    for i in range(len(time_after_15_list)):
        time_after_15_list[i] = int(time_after_15_list[i])
    time_after_15_list
    print(time_after_15_list)   

    Temperature = temp_model.predict([time_after_15_list])[0]
    GHI = ghi_model.predict([time_after_15_list])[0]
    
    print(Temperature, GHI)

    # Solar Power Calculation

    f = 0.18 * 7.4322 * GHI
    insi = Temperature - 25
    midd = 1 - 0.05 * insi 
    power = f * midd

    Power = f * midd
    print(Power)
    
    try:
        cursor.execute("INSERT INTO linear_regression_method (updated_time, temperature, ghi, power) VALUES (%s, %s, %s, %s)", (time_after_15, Temperature, GHI, power))
        print("Insertion Sucessful - LR Method")
        conn.commit()
    except Exception as e:
        print("Error: ", e)
        cursor.close()
        conn.close()
    except KeyboardInterrupt as e2:
        print(e)
        
    time.sleep(5)

# print(predicted_ghis)
# import matplotlib.pyplot as plt
# plt.plot(predicted_temperatures, predicted_ghis, marker='o')
# plt.xlabel('Temperature')
# plt.ylabel('GHI')
# plt.title('Predicted Temperature vs GHI')
# plt.show()

'''