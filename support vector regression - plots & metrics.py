import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions for test set
X_temp_test_scaled = scaler_temp.transform(X_temp_test.reshape(-1, 1)).ravel()
X_temp_pred_scaled = svr_temp.predict(X_test_scaled)
Y_GHI_test_scaled = scaler_GHI.transform(Y_GHI_test.reshape(-1, 1)).ravel()
Y_GHI_pred_scaled = svr_GHI.predict(X_test_scaled)

# Inverse scaling for metrics calculation
X_temp_test_pred = scaler_temp.inverse_transform(X_temp_pred_scaled.reshape(-1, 1)).ravel()
Y_GHI_test_pred = scaler_GHI.inverse_transform(Y_GHI_pred_scaled.reshape(-1, 1)).ravel()

# Metrics calculation
mae_temp = mean_absolute_error(X_temp_test, X_temp_test_pred)
mse_temp = mean_squared_error(X_temp_test, X_temp_test_pred)
rmse_temp = np.sqrt(mse_temp)
r2_temp = r2_score(X_temp_test, X_temp_test_pred)

mae_GHI = mean_absolute_error(Y_GHI_test, Y_GHI_test_pred)
mse_GHI = mean_squared_error(Y_GHI_test, Y_GHI_test_pred)
rmse_GHI = np.sqrt(mse_GHI)
r2_GHI = r2_score(Y_GHI_test, Y_GHI_test_pred)

# Print metrics
print("Temperature Prediction Metrics:")
print(f"Mean Absolute Error (MAE): {mae_temp}")
print(f"Mean Squared Error (MSE): {mse_temp}")
print(f"Root Mean Squared Error (RMSE): {rmse_temp}")
print(f"R-squared (R²): {r2_temp}")

print("\nGHI Prediction Metrics:")
print(f"Mean Absolute Error (MAE): {mae_GHI}")
print(f"Mean Squared Error (MSE): {mse_GHI}")
print(f"Root Mean Squared Error (RMSE): {rmse_GHI}")
print(f"R-squared (R²): {r2_GHI}")

# Visualization
plt.figure(figsize=(10, 6))
sns.kdeplot(X_temp_test, label='Actual Temperature', color='green', fill=True)
sns.kdeplot(X_temp_test_pred, label='Predicted Temperature', color='red', fill=True)
plt.title('KDE: Actual vs Predicted Temperature')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_GHI_test, label='Actual GHI', color='blue', fill=True)
sns.kdeplot(Y_GHI_test_pred, label='Predicted GHI', color='orange', fill=True)
plt.title('KDE: Actual vs Predicted GHI')
plt.xlabel('GHI')
plt.ylabel('Density')
plt.legend()
plt.show()
