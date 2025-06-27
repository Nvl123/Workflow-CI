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
import sys

def main():
    # Argparse untuk menerima parameter dari MLflow Project
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_intercept", type=str, default="True", 
                       help="Whether to calculate the intercept for this model.")
    args = parser.parse_args()
    
    # Convert string to boolean
    fit_intercept = args.fit_intercept.lower() == 'true'
    
    print(f"Starting training with fit_intercept={fit_intercept}")
    
    # Set experiment name
    experiment_name = "Air_Quality_Model"
    
    try:
        mlflow.set_experiment(experiment_name)
        print(f"Using experiment: {experiment_name}")
    except Exception as e:
        print(f"Warning: Could not set experiment {experiment_name}: {e}")
        print("Using default experiment")
    
    print(f"Loading dataset...")
    
    # Look for dataset in current directory and parent directory
    dataset_paths = [
        "air_quality_cleaned.csv",
        "../air_quality_cleaned.csv",
        "data/air_quality_cleaned.csv"
    ]
    
    df = None
    dataset_path = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dataset_path = path
                print(f"Dataset loaded successfully from: {path}")
                print(f"Dataset shape: {df.shape}")
                break
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
                continue
    
    if df is None:
        print("Error: Dataset 'air_quality_cleaned.csv' not found!")
        print("Searched in the following locations:")
        for path in dataset_paths:
            print(f"  - {os.path.abspath(path)} (exists: {os.path.exists(path)})")
        print("\nCurrent directory contents:")
        print(os.listdir("."))
        sys.exit(1)
    
    # Display basic dataset info
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Dataset info:")
    print(df.info())
    
    # Pisahkan fitur dan target
    feature_columns = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
    target_column = 'AQI'
    
    # Validasi kolom
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Check for missing values
    missing_values = df[feature_columns + [target_column]].isnull().sum()
    print(f"Missing values per column:")
    print(missing_values)
    
    if missing_values.sum() > 0:
        print("Warning: Dataset contains missing values. Dropping rows with missing values...")
        df = df.dropna(subset=feature_columns + [target_column])
        print(f"Dataset shape after dropping missing values: {df.shape}")
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Feature columns: {feature_columns}")
    
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
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("feature_columns", ",".join(feature_columns))
        
        # Inisialisasi dan latih model
        model = LinearRegression(fit_intercept=fit_intercept)
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        print("Making predictions...")
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
        
        # Log model coefficients
        if hasattr(model, 'coef_'):
            for i, (feature, coef) in enumerate(zip(feature_columns, model.coef_)):
                mlflow.log_param(f"coef_{feature}", coef)
        
        if hasattr(model, 'intercept_'):
            mlflow.log_param("intercept", model.intercept_)
        
        print("Logging model...")
        # Log model
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None
            )
            print("Model logged successfully to MLflow")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")
        
        # Simpan model sebagai joblib untuk backup
        model_path = "model.pkl"
        try:
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            print("Model saved as joblib artifact")
        except Exception as e:
            print(f"Warning: Could not save model as joblib: {e}")
        
        # Log feature names
        try:
            with open("feature_names.txt", "w") as f:
                f.write("\n".join(feature_columns))
            mlflow.log_artifact("feature_names.txt")
        except Exception as e:
            print(f"Warning: Could not log feature names: {e}")
        
        # Log sample predictions for verification
        try:
            sample_predictions = pd.DataFrame({
                'actual': y_test.head(10).values,
                'predicted': y_pred[:10]
            })
            sample_predictions.to_csv("sample_predictions.csv", index=False)
            mlflow.log_artifact("sample_predictions.csv")
        except Exception as e:
            print(f"Warning: Could not log sample predictions: {e}")
        
        # Print hasil
        print(f"\n=== Training Results ===")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        
        if hasattr(model, 'coef_'):
            print(f"\nModel Coefficients:")
            for feature, coef in zip(feature_columns, model.coef_):
                print(f"  {feature}: {coef:.4f}")
        
        if hasattr(model, 'intercept_'):
            print(f"Intercept: {model.intercept_:.4f}")
        
        # Cleanup temporary files
        temp_files = [model_path, "feature_names.txt", "sample_predictions.csv"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"\nTraining completed successfully!")
        return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("Script executed successfully")
            sys.exit(0)
        else:
            print("Script failed")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
