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

def main():
    # Argparse untuk menerima parameter dari MLflow Project
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_intercept", type=str, default="True", 
                       help="Whether to calculate the intercept for this model.")
    args = parser.parse_args()
    
    # Convert string to boolean
    fit_intercept = args.fit_intercept.lower() == 'true'
    
    # Set experiment name
    experiment_name = "Air_Quality_Model"
    mlflow.set_experiment(experiment_name)
    
    print(f"Loading dataset...")
    
    # Muat dataset
    try:
        df = pd.read_csv("air_quality_cleaned.csv")
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset 'air_quality_cleaned.csv' not found!")
        print("Please ensure the dataset is in the same directory as train.py")
        return
    
    # Pisahkan fitur dan target
    feature_columns = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
    target_column = 'AQI'
    
    # Validasi kolom
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Mulai MLflow run
    with mlflow.start_run() as run:
        print(f"Started MLflow run with ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("fit_intercept", fit_intercept)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        
        # Inisialisasi dan latih model
        model = LinearRegression(fit_intercept=fit_intercept)
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrik
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )
        
        # Simpan model sebagai joblib untuk backup
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Log feature names
        with open("feature_names.txt", "w") as f:
            f.write("\n".join(feature_columns))
        mlflow.log_artifact("feature_names.txt")
        
        # Print hasil
        print(f"Training completed!")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        
        # Cleanup temporary files
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists("feature_names.txt"):
            os.remove("feature_names.txt")

if __name__ == "__main__":
    main()
