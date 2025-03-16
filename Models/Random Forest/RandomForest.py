import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle any zero values to avoid division by zero
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def train_random_forest_model(data_path):
    """
    Train a Random Forest model on the processed housing data with comprehensive metrics.
    
    Parameters:
    data_path (str): Path to the processed CSV file
    
    Returns:
    model (RandomForestRegressor): Trained random forest model
    X_test (DataFrame): Test features
    y_test (Series): Test target values
    metrics (dict): Dictionary of performance metrics
    """
    # Load the processed data
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['resale_price'], errors='ignore')
    y = df['resale_price']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the random forest model
    # These hyperparameters can be tuned for better performance
    model = RandomForestRegressor(
        n_estimators=100,     # Number of trees in the forest
        max_depth=15,         # Maximum depth of the trees
        min_samples_split=5,  # Minimum samples required to split
        n_jobs=-1,            # Use all available cores
        random_state=42
    )
    
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
    # We'll use a smaller number of trees for cross-validation to speed it up
    cv_model = RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_split=5, random_state=42)
    
    cv_r2_scores = cross_val_score(
        cv_model, X, y, cv=5, scoring='r2'
    )
    
    cv_rmse_scores = np.sqrt(-cross_val_score(
        cv_model, X, y, cv=5, scoring='neg_mean_squared_error'
    ))
    
    # Store all metrics in a dictionary
    metrics = {
        'model_name': 'Random Forest Regression',
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
        'cv_rmse_std': cv_rmse_scores.std(),
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
    }
    
    # Print results
    print(f"\nRandom Forest Regression Model Performance:")
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
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Random Forest: Actual vs Predicted Prices')
    plt.savefig('random_forest_actual_vs_predicted.png')
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Random Forest: Residual Plot')
    plt.savefig('random_forest_residuals.png')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances - Random Forest')
    plt.gca().invert_yaxis()  # To have the most important at the top
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Random Forest: Error Distribution')
    plt.savefig('random_forest_error_distribution.png')
    plt.close()
    
    return model, X_test, y_test, metrics

def main():
    # Path to your processed data
    processed_data_path = "../../Dataset/processed_Resaleflatprices.csv"
    
    # Train the model
    model, X_test, y_test, metrics = train_random_forest_model(processed_data_path)
    
    # Save the model
    import joblib
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")
    
    # Save metrics for later comparison
    pd.DataFrame([metrics]).to_csv('random_forest_metrics.csv', index=False)
    print("Metrics saved to 'random_forest_metrics.csv'")

if __name__ == "__main__":
    main()