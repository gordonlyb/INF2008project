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
import os

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

def get_top_100_features():
    """Get the top 100 features from feature importance analysis"""
    try:
        # Try to load from feature importance analysis CSV if it exists
        importance_df = pd.read_csv('feature_analysis/xgboost_importance.csv')
        top_features = importance_df.sort_values('Importance', ascending=False).head(100)['Feature'].tolist()
    except:
        # If file doesn't exist, return None - we'll handle this in load_data
        print("Feature importance file not found. Will determine important features from model.")
        top_features = None
    
    return top_features

def load_data():
    """Load the split datasets and filter to top 100 features"""
    print("Loading datasets...")
    processed_csv = "../../Dataset/processed_Resaleflatprices_XGB.csv"
    train_data, val_data, test_data = split_data(processed_csv)
    
    # Get top 100 features
    top_features = get_top_100_features()
    
    # Split features and target
    X_train = train_data.drop('resale_price', axis=1)
    y_train = train_data['resale_price']
    
    X_val = val_data.drop('resale_price', axis=1)
    y_val = val_data['resale_price']
    
    X_test = test_data.drop('resale_price', axis=1)
    y_test = test_data['resale_price']
    
    # Filter to top 100 features if we have them
    if top_features:
        print(f"Filtering to top 100 features based on importance analysis...")
        X_train = X_train[top_features]
        X_val = X_val[top_features]
        X_test = X_test[top_features]
        print(f"Using {len(top_features)} features")
    else:
        # If we don't have feature importance yet, we'll train with all features
        # later code will extract feature importance
        print(f"Using all {X_train.shape[1]} features")
    
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
    
    # Save to CSV for future use
    os.makedirs('feature_analysis', exist_ok=True)
    feat_importances.to_csv('feature_analysis/xgboost_importance.csv', index=False)
    
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

    # Load data (now with top 100 features)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Train baseline model
    baseline_model, baseline_train_results, baseline_val_results = train_baseline_xgboost(
        X_train, y_train, X_val, y_val)
    
    # If we started with all features, now extract feature importance
    # and refilter to top 100 for subsequent steps
    if X_train.shape[1] > 100:
        feature_importance = plot_feature_importance(baseline_model, X_train)
        # Get top 100 features
        top_features = feature_importance.head(100)['Feature'].tolist()
        
        print("\nRefiltering to top 100 features based on model importance...")
        # Refilter data
        X_train = X_train[top_features]
        X_val = X_val[top_features]
        X_test = X_test[top_features]
        
        print(f"New shapes - Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        # Retrain baseline with top 100 features
        baseline_model, baseline_train_results, baseline_val_results = train_baseline_xgboost(
            X_train, y_train, X_val, y_val)
    else:
        # Still plot feature importance for the current model
        feature_importance = plot_feature_importance(baseline_model, X_train)
        
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Plot predictions
    plot_predictions(y_val, baseline_val_results['predictions'], "Baseline Model: Actual vs Predicted")
    
    # Save baseline model
    joblib.dump(baseline_model, 'baseline_xgboost_model_top100.pkl')
    print("\nBaseline model saved as 'baseline_xgboost_model_top100.pkl'")
    
    # Check if best parameters from previous run exist
    best_params_file = 'best_xgboost_params_top100.txt'
    has_previous_params = os.path.exists(best_params_file)
    
    # Initialize these flags to track what approach we take
    use_previous = False
    run_optimization = False
    
    if has_previous_params:
        print(f"\nFound existing hyperparameters in '{best_params_file}'")
        use_previous = input("Use these hyperparameters instead of running optimization again? (y/n): ").lower() == 'y'
    
    # If not using previous params, ask about running optimization
    if not use_previous:
        run_optimization = input("\nRun hyperparameter optimization? (y/n): ").lower() == 'y'
        
    # Handle the different paths
    if use_previous:
        # Load parameters from file
        best_params = {}
        with open(best_params_file, 'r') as f:
            for line in f:
                if ':' in line:
                    param, value = line.strip().split(':', 1)
                    param = param.strip()
                    value = value.strip()
                    # Convert value to appropriate type
                    if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                        best_params[param] = int(value)
                    else:
                        try:
                            best_params[param] = float(value)
                        except ValueError:
                            best_params[param] = value
        
        print("Loaded hyperparameters:")
        print(best_params)
        
        # Train model with loaded parameters
        optimized_model = xgb.XGBRegressor(**best_params, eval_metric='rmse', random_state=42)
        
        print("\nTraining model with previously optimized hyperparameters...")
        optimized_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        # Evaluate the model
        optimized_train_results = evaluate_model(optimized_model, X_train, y_train, "Training (Optimized)")
        optimized_val_results = evaluate_model(optimized_model, X_val, y_val, "Validation (Optimized)")
        
    elif run_optimization:
        # Optimize model
        optimized_model, optimized_train_results, optimized_val_results, best_params = optimize_xgboost(
            X_train, y_train, X_val, y_val, n_trials=50)
        
        # Save best parameters
        with open(best_params_file, 'w') as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
    else:
        # Use baseline model as final model
        print("\nSkipping optimization. Using baseline model as final model.")
        optimized_model = baseline_model
        optimized_train_results = baseline_train_results
        optimized_val_results = baseline_val_results
    
    # Only save optimized model if we actually performed optimization or loaded previous params
    if run_optimization or use_previous:
        # Save optimized model
        joblib.dump(optimized_model, 'optimized_xgboost_model_top100.pkl')
        print("\nOptimized model saved as 'optimized_xgboost_model_top100.pkl'")
        
        # Plot optimized predictions
        plot_predictions(y_val, optimized_val_results['predictions'], "Optimized Model: Actual vs Predicted")
    
    # Evaluate on test set with final model (either optimized or baseline)
    test_results = evaluate_model(optimized_model, X_test, y_test, "Test Set (Final Evaluation)")
    plot_predictions(y_test, test_results['predictions'], "Test Set: Actual vs Predicted")
    
    print("\nModel training with Top 100 Features completed!")

if __name__ == "__main__":
    main()
