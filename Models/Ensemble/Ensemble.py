import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import time
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle any zero values to avoid division by zero
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

class EnsembleModel:
    """
    Ensemble model that combines Linear Regression, Decision Tree, and Random Forest
    using weighted average.
    """
    def __init__(self, models=None, weights=None):
        self.models = models if models is not None else []
        self.weights = weights if weights is not None else []
        self.model_names = []
    
    def fit(self, X, y):
        """
        Fit all base models and find optimal weights.
        """
        start_time = time.time()
        
        # Split the data for weight optimization
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit the base models
        self.models = [
            LinearRegression(),
            DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
            RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, n_jobs=-1, random_state=42)
        ]
        
        self.model_names = ["Linear Regression", "Decision Tree", "Random Forest"]
        
        print("Training base models...")
        for i, model in enumerate(self.models):
            print(f"Training {self.model_names[i]}...")
            model.fit(X_train, y_train)
        
        # Generate predictions for validation set
        val_predictions = []
        for model in self.models:
            val_predictions.append(model.predict(X_val))
        
        # Convert to numpy arrays for easier manipulation
        val_predictions = np.array(val_predictions)
        
        # Find optimal weights using grid search
        # We'll search through different weight combinations
        print("Finding optimal weights...")
        best_mse = float('inf')
        best_weights = [1/len(self.models)] * len(self.models)
        
        # Define the weight grid to search (step of 0.1 from 0 to 1)
        weight_options = np.linspace(0, 1, 11)
        
        # Since weights must sum to 1, we'll use a grid search approach
        # For 3 models, we need to search through valid combinations of 3 weights
        for w1 in weight_options:
            for w2 in weight_options:
                # Ensure the weights sum to 1
                w3 = 1 - w1 - w2
                if w3 < 0 or w3 > 1:
                    continue
                
                weights = [w1, w2, w3]
                
                # Combine predictions using these weights
                ensemble_preds = np.zeros_like(y_val)
                for i, preds in enumerate(val_predictions):
                    ensemble_preds += weights[i] * preds
                
                # Calculate MSE
                mse = mean_squared_error(y_val, ensemble_preds)
                
                # Update best weights if improvement found
                if mse < best_mse:
                    best_mse = mse
                    best_weights = weights
        
        self.weights = best_weights
        print(f"Optimal weights found: {self.weights}")
        
        # Refit models on the entire dataset
        print("Refitting models on entire dataset...")
        for i, model in enumerate(self.models):
            model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.4f} seconds")
        
        return self, training_time
    
    def predict(self, X):
        """
        Make predictions using the weighted average of base models.
        """
        start_time = time.time()
        
        # Get predictions from each base model
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        # Combine predictions using weights
        ensemble_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            ensemble_preds += self.weights[i] * preds
        
        prediction_time = time.time() - start_time
        
        return ensemble_preds, prediction_time
    
    def save(self, filename):
        """
        Save the ensemble model to a file.
        """
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'model_names': self.model_names
        }
        joblib.dump(model_data, filename)
        print(f"Ensemble model saved to {filename}")

def train_ensemble_model(data_path):
    """
    Train an ensemble model combining Linear Regression, Decision Tree, and Random Forest.
    
    Parameters:
    data_path (str): Path to the processed CSV file
    
    Returns:
    model (EnsembleModel): Trained ensemble model
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
    
    # Create and train the ensemble model
    ensemble = EnsembleModel()
    ensemble, training_time = ensemble.fit(X_train, y_train)
    
    # Make predictions
    y_pred, prediction_time = ensemble.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(y_test)
    p = X_test.shape[1]  # Number of predictors
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Store metrics
    metrics = {
        'model_name': 'Ensemble (LR + DT + RF)',
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'mape': mape,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'weights': ensemble.weights,
        'weights_lr': ensemble.weights[0],
        'weights_dt': ensemble.weights[1],
        'weights_rf': ensemble.weights[2]
    }
    
    # Print results
    print(f"\nEnsemble Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R² Score: {adjusted_r2:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"Model Weights: {dict(zip(ensemble.model_names, ensemble.weights))}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Ensemble Model: Actual vs Predicted Prices')
    plt.savefig(os.path.join(script_dir,'ensemble_actual_vs_predicted.png'))
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Ensemble Model: Residual Plot')
    plt.savefig(os.path.join(script_dir,'ensemble_residuals.png'))
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Ensemble Model: Error Distribution')
    plt.savefig(os.path.join(script_dir,'ensemble_error_distribution.png'))
    plt.close()
    
    # Plot model weights
    plt.figure(figsize=(8, 6))
    plt.bar(ensemble.model_names, ensemble.weights)
    plt.ylabel('Weight')
    plt.title('Ensemble Model: Component Weights')
    plt.ylim(0, 1)
    for i, w in enumerate(ensemble.weights):
        plt.text(i, w + 0.02, f'{w:.2f}', ha='center')
    plt.savefig(os.path.join(script_dir,'ensemble_weights.png'))
    plt.close()
    
    return ensemble, X_test, y_test, metrics

def main():
    # Path to your processed data
    script_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    processed_data_path = os.path.join(project_root, "Dataset", "processed_Resaleflatprices.csv")
    
    # Train the ensemble model
    ensemble, X_test, y_test, metrics = train_ensemble_model(processed_data_path)
    
    # Save the model
    ensemble.save(os.path.join(script_dir, 'ensemble_model.pkl'))

    
    # Save metrics for comparison
    pd.DataFrame([metrics]).to_csv(os.path.join(script_dir, 'ensemble_metrics.csv'), index=False)
    print("Metrics saved to 'ensemble_metrics.csv'")
    
    # Load all metrics for comparison
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.dirname(script_dir)  # Parent directory containing all model folders


        lr_metrics = pd.read_csv(os.path.join(models_dir, 'Linear Regression', 'linear_regression_metrics.csv'))
        dt_metrics = pd.read_csv(os.path.join(models_dir, 'Decision Tree', 'decision_tree_metrics.csv'))
        rf_metrics = pd.read_csv(os.path.join(models_dir, 'Random Forest', 'random_forest_metrics.csv'))
        
        # Combine all metrics
        all_metrics = pd.concat([
            lr_metrics, 
            dt_metrics, 
            rf_metrics, 
            pd.DataFrame([metrics])
        ])
        
        # Select key metrics for comparison
        comparison_metrics = all_metrics[['model_name', 'rmse', 'mae', 'r2', 'mape', 'training_time', 'prediction_time']]
        
        # Save comparison to CSV
        comparison_metrics.to_csv(os.path.join(script_dir, 'model_comparison.csv'), index=False)
        print("Model comparison saved to 'model_comparison.csv'")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE comparison
        plt.subplot(2, 2, 1)
        plt.bar(comparison_metrics['model_name'], comparison_metrics['rmse'])
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Plot R² comparison
        plt.subplot(2, 2, 2)
        plt.bar(comparison_metrics['model_name'], comparison_metrics['r2'])
        plt.title('R² Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Plot MAPE comparison
        plt.subplot(2, 2, 3)
        plt.bar(comparison_metrics['model_name'], comparison_metrics['mape'])
        plt.title('MAPE Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Plot training time comparison
        plt.subplot(2, 2, 4)
        plt.bar(comparison_metrics['model_name'], comparison_metrics['training_time'])
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, 'model_comparison.png'))
        plt.close()
        
        print("Model comparison visualization saved to 'model_comparison.png'")
        
    except FileNotFoundError:
        print("Note: Could not find metrics files for all models. Run all model scripts before comparison.")

if __name__ == "__main__":
    main()