import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle any zero values to avoid division by zero
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def train_linear_regression_model(data_path):
    """
    Train a Linear Regression model on the processed housing data with comprehensive metrics.
    
    Parameters:
    data_path (str): Path to the processed CSV file
    
    Returns:
    model (LinearRegression): Trained linear regression model
    X_test (DataFrame): Test features
    y_test (Series): Test target values
    metrics (dict): Dictionary of performance metrics
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))


    # Load the processed data
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['resale_price'], errors='ignore')
    y = df['resale_price']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the linear regression model
    model = LinearRegression()
    
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate standard metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(y_test)
    p = X_test.shape[1]  # Number of predictors
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Cross-validation
    cv_r2_scores = cross_val_score(
        LinearRegression(), X, y, cv=5, scoring='r2'
    )
    
    cv_rmse_scores = np.sqrt(-cross_val_score(
        LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error'
    ))
    
    # Store all metrics in a dictionary
    metrics = {
        'model_name': 'Linear Regression',
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'mape': mape,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std()
    }
    
    # Print results
    print(f"\nLinear Regression Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R² Score: {adjusted_r2:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"5-Fold CV R² Mean: {cv_r2_scores.mean():.4f} (±{cv_r2_scores.std():.4f})")
    print(f"5-Fold CV RMSE Mean: {cv_rmse_scores.mean():.2f} (±{cv_rmse_scores.std():.2f})")
    
    # Get feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    })
    feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance[['Feature', 'Coefficient']].head(10))
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Actual vs Predicted Prices')
    plt.savefig(os.path.join(script_dir,'linear_regression_actual_vs_predicted.png'))
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Linear Regression: Residual Plot')
    plt.savefig(os.path.join(script_dir,'linear_regression_residuals.png'))
    plt.close()
    
    # Plot feature importance (top 15 coefficients)
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Coefficient'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Coefficients - Linear Regression')
    plt.gca().invert_yaxis()  # To have the highest coefficient at the top
    plt.axvline(x=0, color='k', linestyle='--')  # Add vertical line at 0
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir,'linear_regression_feature_importance.png'))
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Linear Regression: Error Distribution')
    plt.savefig(os.path.join(script_dir,'linear_regression_error_distribution.png'))
    plt.close()
    
    # Q-Q plot to check for normality of residuals
    from scipy import stats
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Linear Regression: Q-Q Plot of Residuals')
    plt.savefig(os.path.join(script_dir,'linear_regression_qq_plot.png'))
    plt.close()
    
    return model, X_test, y_test, metrics

def main():
    # Get the absolute directory path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    processed_data_path = os.path.join(project_root, "Dataset", "processed_Resaleflatprices.csv")
    
    # Train the model
    model, X_test, y_test, metrics = train_linear_regression_model(processed_data_path)
    
    # Save the model in the same directory as the script
    import joblib
    joblib.dump(model, os.path.join(script_dir, 'linear_regression_model.pkl'))
    print(f"Model saved to: {os.path.join(script_dir, 'linear_regression_model.pkl')}")
    
    # Save metrics to the same directory as the script
    pd.DataFrame([metrics]).to_csv(os.path.join(script_dir, 'linear_regression_metrics.csv'), index=False)
    print(f"Metrics saved to: {os.path.join(script_dir, 'linear_regression_metrics.csv')}")

if __name__ == "__main__":
    main()