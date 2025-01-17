## Solarpanel Power Prediction using ML regression models :

Real-time ML-powered system designed to predict solar panel power output using multiple regression models. The system uses historical weather data from the NSRDB (https://nsrdb.nrel.gov/data-viewer), specific to your locality, to provide accurate and localized solar power forecasts. By analyzing key weather parameters like Temperature and Global Horizontal Irradiance (GHI), it ensures that the predictions reflect real-time weather conditions, delivering precise solar power generation estimates.

The system integrates three different regression models: Linear Regression, Random Forest, and Support Vector Regression (SVR).

### **System Architecture**  

![architecture](https://github.com/user-attachments/assets/832f286f-1693-492d-8205-6a1ff8fd15f5)

 **My Locality - Random Forest Method - Grafana Dashboard**   <br><br>
<img src="predictions_results/My%20Locality%20-%20Grafana%20Dashboard%20.png" height="300">

---

## Installation Instructions

### Prerequisites
- **Python 3.11+**
- **PostgreSQL**
- **Grafana**

Install all packages using this : ```pip install numpy pandas scikit-learn matplotlib psycopg2```

### Step 1: Clone the Repository
```bash
git clone https://github.com/g-wtham/solar_power_prediction.git
cd solar_power_prediction
```

### Step 2: Set Up PostgreSQL
1. Create a PostgreSQL database:
    ```sql
    CREATE DATABASE solar_power_prediction;
    ```
2. Configure the database tables with the respective table names as per the in the `random_forest.py` file given.

### Step 3: Connect Python to Postgres using Psycopg2 package
1. Define postgres database username and password (default-username: postgress; password: root)
2. psycopg2 is used for postgres to python connection

### Step 4: Install Grafana and select the data source (select postgresql)
1. Install Grafana (https://grafana.com/grafana/download) and set up username and password (default username & password: admin)
2. Navigate to `locahost:3000`, select postgresql as the data source and build dashboards by selecting the corresponding tables from the connected pg database.
3. Toggle the `order` setting 'ON' and set _custom limit_, as default is 50 and can hinder if more data points are plotted.
4. You can export the dashboards as JSON files as well for preserving the templates for external sharing.

### Step 5: Run the System
1. Train the model:
   ```bash
   python random_forest.py 
   ```
2. For getting the plots & metrics for the model:
   ```bash
   python random_forest-metrics.py
   ```
3. Visualize Results in Grafana:
View the predictions on the Grafana dashboard (`localhost:3000`)

---

## Prediction results visualized from various models, showcasing the actual vs. predicted values based on input parameters like GHI and Temperature.

Random Forest method performed better than linear regression and support regression model, as it captures the non-linearity of the dataset well. As the dataset contains more than 43,849 rows, SVR model struggles without performing additional preprocessing steps, while random forest achieves nearly 97.5% accuracy as R² score is 0.9755. Thus, out of 3 regression models, random forest gives us good performance to accuracy ratio.

### Random Forest Metrics :
Mean Absolute Error (MAE): 0.5528913519430765
Mean Squared Error (MSE): 0.5644339387885424
Root Mean Squared Error (RMSE): 0.7512881862431635
R-squared (R²): 0.9755113005363533 

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




