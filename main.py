# =============================================================================
# Analisis & Prediksi Timbulan Sampah dengan AI
# Author: Faisal Dino Bahtiar
# Dataset: SIPSN KLHK (Sistem Informasi Pengelolaan Sampah Nasional)
# =============================================================================

#import library
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import plotly.io as pio


import os
# load dataset
file_path = "data/Data_Timbulan_Sampah_SIPSN_KLHK.xlsx"
if not os.path.exists(file_path):
    file_path = "/kaggle/input/data-timbunan-sampah-di-indonesia/Data_Timbulan_Sampah_SIPSN_KLHK.xlsx"

# Membaca dataset
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Sheet1")


# Mengatur header yang benar
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)


# Mengganti nama kolom agar lebih rapi
df.columns = ["Tahun", "Provinsi", "Kabupaten/Kota", "Timbulan Sampah Harian (ton)", "Timbulan Sampah Tahunan (ton)"]


# Menghapus baris pertama yang berisi string header lama
df = df[1:].reset_index(drop=True)


# Menghitung jumlah nilai null
print("Jumlah nilai null per kolom:\n", df.isnull().sum())


# Mengonversi kolom numerik ke tipe data float
df["Timbulan Sampah Harian (ton)"] = df["Timbulan Sampah Harian (ton)"].astype(float)
df["Timbulan Sampah Tahunan (ton)"] = df["Timbulan Sampah Tahunan (ton)"].astype(float)


# Feature Engineering
df["Rasio Harian/Tahunan"] = df["Timbulan Sampah Harian (ton)"] / df["Timbulan Sampah Tahunan (ton)"]


# Exploratory Data Analysis (EDA)
print("Statistik Deskriptif:\n", df.describe())
print("Informasi Dataset:\n", df.info())


# Preprocessing Data
scaler = StandardScaler()
df[["Timbulan Sampah Harian (ton)", "Timbulan Sampah Tahunan (ton)", "Rasio Harian/Tahunan"]] = scaler.fit_transform(df[["Timbulan Sampah Harian (ton)", "Timbulan Sampah Tahunan (ton)", "Rasio Harian/Tahunan"]])


# PCA untuk Reduksi Dimensi
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[["Timbulan Sampah Harian (ton)", "Timbulan Sampah Tahunan (ton)"]])


# Clustering Analysis
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_pca)


# Model Machine Learning
X = df[["Timbulan Sampah Harian (ton)", "Rasio Harian/Tahunan"]]
y = df["Timbulan Sampah Tahunan (ton)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter Tuning untuk XGBoost
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_


# Model Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
gb_model.fit(X_train, y_train)


# Stacking Model
stacking_model = StackingRegressor(
    estimators=[('xgb', best_xgb_model), ('gb', gb_model)],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train, y_train)


# Model LSTM
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
X_train_lstm = np.expand_dims(X_train, axis=-1)
X_test_lstm = np.expand_dims(X_test, axis=-1)
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, validation_data=(X_test_lstm, y_test))


# Evaluasi Model
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} -> MAE: {mae}, MSE: {mse}, R2 Score: {r2}")
    
    # Visualisasi Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Distribusi Residual {model_name}")
    plt.show()


# Cross-validation score
models = {"Optimized XGBoost": best_xgb_model, "Gradient Boosting": gb_model, "Stacking Model": stacking_model}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name} Cross-Validation R2: {scores.mean():.4f} (+/- {scores.std():.4f})")


evaluate_model(y_test, best_xgb_model.predict(X_test), "Optimized XGBoost")
evaluate_model(y_test, gb_model.predict(X_test), "Gradient Boosting")
evaluate_model(y_test, stacking_model.predict(X_test), "Stacking Model")


# Time Series Forecasting dengan ARIMA
arima_model = ARIMA(y, order=(5,1,0))
arima_result = arima_model.fit()
print(arima_result.summary())


# Visualisasi ACF dan PACF
fig, ax = plt.subplots(1, 2, figsize=(12,5))
plot_acf(y, ax=ax[0])
plot_pacf(y, ax=ax[1])
plt.show()


# Visualisasi Heatmap Korelasi
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap Korelasi Fitur")
plt.show()


# Visualisasi Distribusi Sampah per Provinsi
fig_provinsi = px.bar(df, x='Provinsi', y='Timbulan Sampah Tahunan (ton)', title='Distribusi Timbulan Sampah per Provinsi', color='Provinsi')
fig_provinsi.show()


# Visualisasi Clustering
fig_cluster = px.scatter(df, x="Timbulan Sampah Harian (ton)", y="Timbulan Sampah Tahunan (ton)", color="Cluster", title="Clustering Timbulan Sampah", color_discrete_sequence=px.colors.qualitative.Set1)
fig_cluster.show()


# Visualisasi Prediksi vs Aktual
fig_pred_rf = go.Figure()
fig_pred_rf.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='markers', name='Actual'))
fig_pred_rf.add_trace(go.Scatter(x=y_test.index, y=best_xgb_model.predict(X_test), mode='lines', name='Predicted (Optimized RF)'))
fig_pred_rf.update_layout(title='Prediksi vs Aktual Timbulan Sampah Tahunan (Random Forest)', xaxis_title='Index', yaxis_title='Timbulan Sampah (scaled)')
fig_pred_rf.show()


# Bar Chart Kontribusi Cluster
df_cluster_group = df.groupby("Cluster")["Timbulan Sampah Tahunan (ton)"].sum().reset_index()
fig_cluster_bar = px.bar(df_cluster_group, x="Cluster", y="Timbulan Sampah Tahunan (ton)", title="Kontribusi Cluster dalam Timbulan Sampah", color="Timbulan Sampah Tahunan (ton)", color_continuous_scale="Blues")
fig_cluster_bar.show()

