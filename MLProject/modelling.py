import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os
import argparse
import shutil

# Set tracking URI ke server MLflow menggunakan file-based URI
mlflow.set_tracking_uri("file:/tmp/mlruns")  # Gunakan lokasi berbasis file

# Argparse untuk menerima parameter dari MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--fit_intercept", type=bool, default=True, help="Whether to calculate the intercept for this model.")
args = parser.parse_args()

# Muat dataset
df = pd.read_csv("air_quality_cleaned.csv")

# Pisahkan fitur dan target
X = df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]  # Fitur
y = df['AQI']  # Target

# Split dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model regresi linear dengan parameter yang diterima
model = LinearRegression(fit_intercept=args.fit_intercept)

# Aktifkan MLflow autologging untuk scikit-learn
mlflow.sklearn.autolog()

# Direktori untuk menyimpan artifact
artifact_dir = "/tmp/model_artifacts"  # Ubah direktori ke /tmp

# Pastikan direktori ada
os.makedirs(artifact_dir, exist_ok=True)

# Memeriksa dan menyalin file-file artefak (conda.yaml, python_env.yaml, requirements.txt)
def copy_file_to_artifact(file_path, artifact_dir):
    if os.path.exists(file_path):
        shutil.copy2(file_path, os.path.join(artifact_dir, os.path.basename(file_path)))
        print(f"File {file_path} berhasil disalin ke {artifact_dir}")
    else:
        print(f"File {file_path} tidak ditemukan!")

# Pastikan file conda.yaml, python_env.yaml, dan requirements.txt ada
copy_file_to_artifact("MLProject/conda.yaml", artifact_dir)
copy_file_to_artifact("MLProject/python_env.yaml", artifact_dir)
copy_file_to_artifact("MLProject/requirements.txt", artifact_dir)

# Mulai eksperimen MLflow
with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi model
    y_pred = model.predict(X_test)
    
    # Menghitung metrik-metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    # Log metrik
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)

    # Simpan model sebagai file .pkl
    model_pkl_path = os.path.join(artifact_dir, "model.pkl")
    joblib.dump(model, model_pkl_path)

    # Log file-file artifact
    mlflow.log_artifact(os.path.join(artifact_dir, "conda.yaml"))
    mlflow.log_artifact(os.path.join(artifact_dir, "python_env.yaml"))
    mlflow.log_artifact(os.path.join(artifact_dir, "requirements.txt"))
    mlflow.log_artifact(model_pkl_path)

    # Simpan model ke dalam format MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"Training selesai dan model disimpan sebagai artifact.")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"RMSE: {rmse}")
