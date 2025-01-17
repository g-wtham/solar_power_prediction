import psycopg2 as postgres
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import datetime
import time

# Database connection
conn = postgres.connect(database="solar_power_prediction", user="postgres", password="root")
cursor = conn.cursor()

# Load dataset
path = "./Temperature_&_GHI_ 2017_2020_myLocality_cleaned.csv"
df = pd.read_csv(path)

# Data Cleaning
df = df.fillna(0)
nonzero_mean = df[df != 0].mean()

# Features and target
cols = [0, 1, 2, 3, 4]
X_input = df[df.columns[cols]].values

Y_temp = df[df.columns[5]].values
Y_GHI = df[df.columns[6]].values

# Train-test split
X_train, X_test, X_temp_train, X_temp_test = train_test_split(X_input, Y_temp, random_state=32)
_, _, Y_GHI_train, Y_GHI_test = train_test_split(X_input, Y_GHI, random_state=32)

# Standardize the features for SVR
scaler_X = StandardScaler()
scaler_temp = StandardScaler()
scaler_GHI = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_temp_train_scaled = scaler_temp.fit_transform(X_temp_train.reshape(-1, 1)).ravel()
Y_GHI_train_scaled = scaler_GHI.fit_transform(Y_GHI_train.reshape(-1, 1)).ravel()

# Initialize SVR models
svr_temp = SVR(kernel='rbf')
svr_GHI = SVR(kernel='rbf')

# Train SVR models
svr_temp.fit(X_train_scaled, X_temp_train_scaled)
svr_GHI.fit(X_train_scaled, Y_GHI_train_scaled)

# Solar Power Prediction
current_time = datetime.datetime.now()
initial_time = current_time

while True:
    # Predicting new values
    initial_time = initial_time + datetime.timedelta(minutes=10)
    copy_time_after_15 = initial_time

    print(initial_time)

    time_after_15 = initial_time.strftime("%Y-%m-%d %H:%M")
    print(time_after_15)

    # Split time into features
    time_after_15_list = copy_time_after_15.strftime("%Y,%m,%d,%H,%M").split(',')
    
    # Predict Temperature and GHI
    Temperature_scaled = svr_temp.predict([time_after_15_list])
    GHI_scaled = svr_GHI.predict([time_after_15_list])

    Temperature = scaler_temp.inverse_transform([Temperature_scaled])[0].item()
    GHI = scaler_GHI.inverse_transform([GHI_scaled])[0].item()

    print(f"Temperature: {Temperature}, GHI: {GHI}")
    
    float(Temperature)
    float()

    # Solar Power Calculation
    f = 0.18 * 7.4322 * GHI
    insi = Temperature - 25
    midd = 1 - 0.05 * insi
    Power = f * midd

    print(f"Power: {Power}")

    # Insert results into database
    try:
        cursor.execute(
            "INSERT INTO svm_method (updated_current_time, temperature, ghi, power) VALUES (%s, %s, %s, %s)",
            (time_after_15, float(Temperature), float(GHI), float(Power))
        )
        print("Insertion Success.")
        conn.commit()
    except Exception as e:
        print("Error Occurred:", e)

    time.sleep(5)
