import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import optuna  # For hyperparameter optimization
import joblib  # For saving models
from sklearn.model_selection import train_test_split

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def split_data(csv_path, target_col='resale_price', test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into train, validation and test sets and save to separate CSV files
    
    Parameters:
    -----------
    csv_path : str
        Path to the processed CSV file
    target_col : str, default='resale_price'
        The name of the target column
    test_size : float, default=0.2
        Proportion of data to use for testing
    val_size : float, default=0.25
        Proportion of remaining data (after test split) to use for validation
    random_state : int, default=42
        Seed for reproducible splits
    """
    # Load the processed data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    print(f"Full dataset shape: {df.shape}")
    
    # First split off the test set
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Then split the train_val into train and validation
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)
    
    return train, val, test

def load_data():
    """Load the split datasets"""
    print("Loading datasets...")
    processed_csv = "../../Dataset/processed_Resaleflatprices_XGB.csv"
    train_data, val_data, test_data = split_data(processed_csv)

    # Split features and target
    X_train = train_data.drop('resale_price', axis=1)
    y_train = train_data['resale_price']
    
    X_val = val_data.drop('resale_price', axis=1)
    y_val = val_data['resale_price']
    
    X_test = test_data.drop('resale_price', axis=1)
    y_test = test_data['resale_price']
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model performance"""
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X)
    pred_time = time.time() - start_time
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # Print metrics
    print(f"\n{dataset_name} Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Prediction Time: {pred_time:.4f} seconds")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'pred_time': pred_time,
        'predictions': y_pred
    }

def train_baseline_xgboost(X_train, y_train, X_val, y_val):
    """Train a baseline XGBoost model with default hyperparameters"""
    print("\n" + "="*50)
    print("Training Baseline XGBoost Model")
    print("="*50)
    
    # Set up basic parameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse',
        'random_state': 42
    }
    
    # Create and train model
    start_time = time.time()
    model = xgb.XGBRegressor(**params)
    
    # For newer XGBoost versions, use callbacks for early stopping
    callbacks = [xgb.callback.EarlyStopping(rounds=20)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    train_time = time.time() - start_time
    
    print(f"Training Time: {train_time:.2f} seconds")
    
    # Rest of the function remains the same
    train_results = evaluate_model(model, X_train, y_train, "Training")
    val_results = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_results, val_results

def plot_feature_importance(model, X_train):
    """Plot feature importance from the model"""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    feat_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    })
    
    # Sort by importance
    feat_importances = feat_importances.sort_values('Importance', ascending=False)
    
    # Display top 20 features
    top_features = feat_importances.head(20)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return feat_importances

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Prices"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

def objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for Optuna optimization"""
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'eval_metric': 'rmse',
        'random_state': 42
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    # For newer XGBoost versions, use callbacks for early stopping
    callbacks = [xgb.callback.EarlyStopping(rounds=20)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Rest of function remains the same
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=100):
    """Optimize XGBoost using Bayesian optimization with Optuna"""
    print("\n" + "="*50)
    print("Optimizing XGBoost Hyperparameters")
    print("="*50)
    
    # Create study object
    study = optuna.create_study(direction='minimize')
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"Best RMSE: {best_value:.4f}")
    print("Best hyperparameters:", best_params)
    
    # Train model with best parameters
    best_model = xgb.XGBRegressor(**best_params, eval_metric='rmse', random_state=42)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # Remove eval_metric='rmse' from here
        verbose=True
    )
    
    # Evaluate optimized model
    train_results = evaluate_model(best_model, X_train, y_train, "Training (Optimized)")
    val_results = evaluate_model(best_model, X_val, y_val, "Validation (Optimized)")
    
    return best_model, train_results, val_results, best_params

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Train baseline model
    baseline_model, baseline_train_results, baseline_val_results = train_baseline_xgboost(
        X_train, y_train, X_val, y_val)
    
    # Plot feature importance
    feature_importance = plot_feature_importance(baseline_model, X_train)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Plot predictions
    plot_predictions(y_val, baseline_val_results['predictions'], "Baseline Model: Actual vs Predicted")
    
    # Save baseline model
    joblib.dump(baseline_model, 'baseline_xgboost_model.pkl')
    print("\nBaseline model saved as 'baseline_xgboost_model.pkl'")
    
    # Ask if user wants to run hyperparameter optimization
    run_optimization = input("\nRun hyperparameter optimization? (y/n): ").lower() == 'y'
    
    if run_optimization:
        # Optimize model
        optimized_model, optimized_train_results, optimized_val_results, best_params = optimize_xgboost(
            X_train, y_train, X_val, y_val, n_trials=50)
        
        # Plot optimized predictions
        plot_predictions(y_val, optimized_val_results['predictions'], "Optimized Model: Actual vs Predicted")
        
        # Save optimized model
        joblib.dump(optimized_model, 'optimized_xgboost_model.pkl')
        print("\nOptimized model saved as 'optimized_xgboost_model.pkl'")
        
        # Save best parameters
        with open('best_xgboost_params.txt', 'w') as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
        
        # Evaluate on test set
        test_results = evaluate_model(optimized_model, X_test, y_test, "Test Set (Final Evaluation)")
        plot_predictions(y_test, test_results['predictions'], "Test Set: Actual vs Predicted")
    else:
        # Evaluate baseline on test set
        test_results = evaluate_model(baseline_model, X_test, y_test, "Test Set (Baseline Model)")
        plot_predictions(y_test, test_results['predictions'], "Test Set: Actual vs Predicted")
    
    print("\nStep 2 (Single Model Optimization) completed!")

if __name__ == "__main__":
    main()
