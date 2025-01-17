import psycopg2 as postgres
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

current_time = datetime.datetime.now()
initial_time = current_time

# Solar Power Prediction
while True:
    # Predicting new values after the model is trained (fit).

    initial_time = initial_time + datetime.timedelta(minutes = 10) # Calcs what happens every 15 mins, as 1 sec passes.
    copy_time_after_15 = initial_time

    print(initial_time)

    time_after_15 = initial_time.strftime("%Y-%m-%d %H:%M")
    print(time_after_15)

    # As models like Random Forest Regressor, expect input data as arrays as individual features, thus we can also do better feature analysis later..
    # Splitting the time into individual elements further as year, month, hour, and second..
    # Before it would be JUST one formatted 'TIME & DATE' string, now it becomes an array list of features than a single feature.

    time_after_15_list = copy_time_after_15.strftime("%Y,%m,%d,%H,%M") # Using a copy instead of recalculating time again..
    time_after_15_list = time_after_15_list.split(',')

    time_after_15_list

    print(time_after_15_list)
    type(time_after_15_list[0])

    Temperature = Rfr1.predict([time_after_15_list])[0]
    GHI = Rfr2.predict([time_after_15_list])[0]
    
    print(Temperature, GHI)

    # Solar Power Calculation

    f = 0.18 * 7.4322 * GHI
    insi = Temperature - 25
    midd = 1 - 0.05 * insi 
    power = f * midd

    Power = f * midd
    print(Power)
    
    try:
        cursor.execute("INSERT INTO rfm_locality (updated_current_time, temperature, ghi, power) VALUES (%s, %s, %s, %s)", (time_after_15, Temperature, GHI, Power))
        print("Insertion Success.")
        conn.commit()
    except Exception as e:
        print("Error Occured.", e)
        cursor.close()
        conn.close()
        
    time.sleep(5)