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
mlflow.set_tracking_uri("file:/tmp/mlruns")

# Buat experiment jika belum ada
try:
    experiment_name = "Air_Quality_Model"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error creating experiment: {e}")
    print("Using default experiment")

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

# JANGAN gunakan autolog untuk menghindari konflik
# mlflow.sklearn.autolog()  # Disable autolog

# Direktori untuk menyimpan artifact
artifact_dir = "/tmp/model_artifacts"

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
copy_file_to_artifact("conda.yaml", artifact_dir)
copy_file_to_artifact("python_env.yaml", artifact_dir) 
copy_file_to_artifact("requirements.txt", artifact_dir)

# Mulai eksperimen MLflow
with mlflow.start_run() as run:
    print(f"Started MLflow run with ID: {run.info.run_id}")
    
    # Log parameters
    mlflow.log_param("fit_intercept", args.fit_intercept)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi model
    y_pred = model.predict(X_test)
    
    # Menghitung metrik-metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Log metrik
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    
    # Log dataset info
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", X.shape[0])
    mlflow.log_param("n_train", X_train.shape[0])
    mlflow.log_param("n_test", X_test.shape[0])
    
    # Simpan model sebagai file .pkl
    model_pkl_path = os.path.join(artifact_dir, "model.pkl")
    joblib.dump(model, model_pkl_path)
    
    # Log file-file artifact jika ada
    for filename in ["conda.yaml", "python_env.yaml", "requirements.txt"]:
        filepath = os.path.join(artifact_dir, filename)
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath)
    
    # Log model pkl
    if os.path.exists(model_pkl_path):
        mlflow.log_artifact(model_pkl_path)
    
    # Simpan model ke dalam format MLflow (manual logging)
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )
        print("Model berhasil disimpan ke MLflow")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
        # Tetap lanjutkan eksekusi
    
    print(f"Training selesai dan model disimpan sebagai artifact.")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
