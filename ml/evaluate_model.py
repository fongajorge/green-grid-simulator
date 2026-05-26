import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'simulator', 'src')
sys.path.append(src_path)

from ml_model import linear_regression, scaler

# ---------------------------------------------------------
# MATHEMATICAL FORMULAS
# ---------------------------------------------------------
def calculate_mse(y_true, y_pred):
    """Mean Squared Error: Average of the squared errors."""
    return np.mean((y_true - y_pred) ** 2)

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error: Average of the absolute errors."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_r2(y_true, y_pred):
    """R-squared: How much variance in y is explained by the model."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def run_evaluation():
    print("Loading cleaned data...")
    df = pd.read_csv('clean_dataset.csv')
    y = df['Solar_Generation_kW'].values
    
    # ---------------------------------------------------------
    # 1. TRAIN MODEL A (Three variables: Temp, Humidity, Irradiance)
    # ---------------------------------------------------------
    features_A = ['Temperature_C', 'Humidity_percent', 'Irradiance_Wm2']
    X_A = df[features_A].values
    
    scaler_A = scaler()
    X_A_scaled = scaler_A.fit_transform(X_A)
    
    model_A = linear_regression(learning_rate=0.01, iterations=1500)
    model_A.fit(X_A_scaled, y)
    y_pred_A = model_A.predict(X_A_scaled)

    # ---------------------------------------------------------
    # 2. TRAIN MODEL B (Baseline: Irradiance only)
    # ---------------------------------------------------------
    features_B = ['Irradiance_Wm2']
    X_B = df[features_B].values
    
    scaler_B = scaler()
    X_B_scaled = scaler_B.fit_transform(X_B)
    
    model_B = linear_regression(learning_rate=0.01, iterations=1500)
    model_B.fit(X_B_scaled, y)
    y_pred_B = model_B.predict(X_B_scaled)

    # ---------------------------------------------------------
    # 3. RESULTS
    # ---------------------------------------------------------
    print("\n--- Model A Results (Multivariate) ---")
    print(f"MSE: {calculate_mse(y, y_pred_A):.2f}")
    print(f"MAE: {calculate_mae(y, y_pred_A):.2f}")
    print(f"R^2: {calculate_r2(y, y_pred_A):.4f}")

    print("\n--- Model B Results (Single Variable) ---")
    print(f"MSE: {calculate_mse(y, y_pred_B):.2f}")
    print(f"MAE: {calculate_mae(y, y_pred_B):.2f}")
    print(f"R^2: {calculate_r2(y, y_pred_B):.4f}")

    # ---------------------------------------------------------
    # 4. DATA VISUALIZATION (PNG Generation with Timestamps)
    # ---------------------------------------------------------
    from datetime import datetime
    
    print("\nGenerating charts for the report...")
    
    # Generate the current timestamp string (e.g., "2026-05-25_21-47")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Dynamically create the ml/outputs/ folder if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Chart 1: Raw Data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Irradiance_Wm2'], df['Solar_Generation_kW'], alpha=0.3, color='orange', label='Real Observations')
    plt.title('Relationship: Solar Irradiance vs Energy Generation')
    plt.xlabel('Irradiance (W/m^2)')
    plt.ylabel('Solar Generation (kW)')
    plt.grid(True)
    
    # Save to the new outputs folder
    filename_1 = f'raw_data_plot_{timestamp}.png'
    filepath_1 = os.path.join(outputs_dir, filename_1)
    plt.savefig(filepath_1)
    plt.close()

    # Chart 2: Reality vs Model A Prediction
    plt.figure(figsize=(12, 5))
    plt.plot(y[:168], label='Real Generation', color='black', linewidth=2) # 168 hours = 1 week
    plt.plot(y_pred_A[:168], label='Model A Prediction', color='blue', linestyle='--')
    plt.title('Digital Twin Performance: Reality vs Prediction (1 Week Sample)')
    plt.xlabel('Elapsed Hours')
    plt.ylabel('Solar Generation (kW)')
    plt.legend()
    plt.grid(True)
    
    # Save to the new outputs folder
    filename_2 = f'prediction_vs_reality_{timestamp}.png'
    filepath_2 = os.path.join(outputs_dir, filename_2)
    plt.savefig(filepath_2)
    plt.close()

    print(f"Success! Charts saved to the 'ml/outputs/' folder as '{filename_1}' and '{filename_2}'.")

if __name__ == "__main__":
    run_evaluation()
