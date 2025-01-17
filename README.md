## Solar Power Prediction using ML regression models :

Real-time ML-powered system designed to predict solar panel power output using multiple regression models. The system uses historical weather data from the NSRDB (https://nsrdb.nrel.gov/data-viewer), specific to your locality, to provide accurate and localized solar power forecasts. By analyzing key weather parameters like Temperature and Global Horizontal Irradiance (GHI), it ensures that the predictions reflect real-time weather conditions, delivering precise solar power generation estimates.

The system integrates three different regression models: Linear Regression, Random Forest, and Support Vector Regression (SVR).

### **System Architecture**  

![architecture](https://github.com/user-attachments/assets/832f286f-1693-492d-8205-6a1ff8fd15f5)

 **My Locality - Random Forest Method - Grafana Dashboard**   <br><br>
<img src="predictions_results/My%20Locality%20-%20Grafana%20Dashboard%20.png" height="300">

## Prediction results visualized from various models, showcasing the actual vs. predicted values based on input parameters like GHI and Temperature.

### **1. Linear Regression**

- **Actual vs Predicted GHI**  
  <img src="predictions_results/linear_regression_actual_vs_pred_GHI.png" height="300">
  
- **Actual vs Predicted Temperature**  
  <img src="predictions_results/linear_regression_actual_vs_pred_temp.png" height="300">

### **2. Support Vector Regression (SVR)**

- **Actual vs Predicted GHI**  
  <img src="predictions_results/SVR_actual_vs_pred_GHI.png" height="300">

- **Actual vs Predicted Temperature**  
  <img src="predictions_results/SVR_actual_vs_pred_temp.png" height="300">

### **3. Random Forest**

- **Actual vs Predicted GHI**  
  <img src="predictions_results/random_forest_actual_vs_pred_GHI.png" height="300">

- **Actual vs Predicted Temperature**  
  <img src="predictions_results/random_forest_actual_vs_pred_temp.png" height="300">

### **4. Grafana Dashboard**

 **All methods dashboard** <br>
  <img src="predictions_results/Grafana Dashboard .png" height="500">




