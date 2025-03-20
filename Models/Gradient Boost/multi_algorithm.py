import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import the algorithms
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_data_with_top_features(data_path, top_features_count=100):
    """
    Load data and filter to only use top features
    """
    print(f"Loading data from {data_path}...")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Split into train/test first to maintain consistency with previous steps
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)
    
    # Get target column
    target_col = 'resale_price'
    
    # Check if feature importance file exists
    feature_file = 'feature_analysis/xgboost_importance.csv'
    if os.path.exists(feature_file):
        print(f"Loading top {top_features_count} features from {feature_file}")
        features_df = pd.read_csv(feature_file)
        top_features = features_df.sort_values('Importance', ascending=False).head(top_features_count)['Feature'].tolist()
        
        # Make sure all features exist in the dataset
        top_features = [f for f in top_features if f in df.columns]
        
        # Ensure target column is not in features
        if target_col in top_features:
            top_features.remove(target_col)
            
        print(f"Using {len(top_features)} features")
    else:
        print("Feature importance file not found. Using all features except target.")
        top_features = [col for col in df.columns if col != target_col]
    
    # Split into features and target
    X_train = train[top_features]
    y_train = train[target_col]
    
    X_val = val[top_features]
    y_val = val[target_col]
    
    X_test = test[top_features]
    y_test = test[target_col]
    
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

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Prices", filename=None):
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
    
    if filename:
        plt.savefig(f"results/{filename}.png")
    
    plt.close()

def compare_algorithms(results_dict, metric='rmse', dataset='val'):
    """Compare algorithms using the specified metric"""
    algorithms = list(results_dict.keys())
    metrics = [results_dict[alg][dataset][metric] for alg in algorithms]
    
    # For R², higher is better. For others, lower is better
    if metric == 'r2':
        best_idx = np.argmax(metrics)
    else:
        best_idx = np.argmin(metrics)
        
    best_alg = algorithms[best_idx]
    best_value = metrics[best_idx]
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    
    # For RMSE, MAPE, MAE, lower is better
    if metric != 'r2':
        colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(algorithms))]
        plt.bar(algorithms, metrics, color=colors)
        plt.ylabel(f"{metric.upper()}")
        plt.title(f"Algorithm Comparison by {metric.upper()} (Lower is Better)")
    else:
        colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(algorithms))]
        plt.bar(algorithms, metrics, color=colors)
        plt.ylabel(f"{metric.upper()}")
        plt.title(f"Algorithm Comparison by {metric.upper()} (Higher is Better)")
        
    # Add value labels on top of each bar
    for i, v in enumerate(metrics):
        if metric == 'mape':
            plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        elif metric == 'r2':
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        else:
            plt.text(i, v + (max(metrics) * 0.02), f"${v:.2f}" if 'mae' in metric or 'rmse' in metric else f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig(f"results/comparison_{metric}.png")
    plt.close()
    
    print(f"\nBest algorithm by {metric.upper()}: {best_alg} with {metric.upper()} = ", end="")
    if metric == 'mape':
        print(f"{best_value:.2f}%")
    elif metric == 'r2':
        print(f"{best_value:.4f}")
    else:
        print(f"${best_value:.2f}" if 'mae' in metric or 'rmse' in metric else f"{best_value:.4f}")
        
    return best_alg, best_value

# Function to load parameters from file
def load_params_from_file(file_path):
    """Load hyperparameters from a text file"""
    if not os.path.exists(file_path):
        return None
        
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                param, value = line.strip().split(':', 1)
                param = param.strip()
                value = value.strip()
                
                # Convert value to appropriate type
                try:
                    # Try as int first
                    params[param] = int(value)
                except ValueError:
                    try:
                        # Then as float
                        params[param] = float(value)
                    except ValueError:
                        # Finally as string
                        params[param] = value
    
    return params

# XGBoost Optimization Function
def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50, use_previous=False):
    """Optimize XGBoost using Bayesian optimization or load previous parameters"""
    params_file = 'models/xgboost_params.txt'
    
    # Check if we should use previous parameters
    if use_previous and os.path.exists(params_file):
        print("\n" + "="*50)
        print("Loading previous XGBoost hyperparameters")
        print("="*50)
        
        best_params = load_params_from_file(params_file)
        print("Loaded parameters:", best_params)
        
        # Train model with loaded parameters
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
    else:
        print("\n" + "="*50)
        print("Optimizing XGBoost")
        print("="*50)
        
        def objective(trial):
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
                'random_state': 42,
                # Add early_stopping_rounds here instead of in fit()
                'early_stopping_rounds': 50
            }
            
            # Remove early_stopping_rounds from params before creating model
            early_stopping_rounds = params.pop('early_stopping_rounds')
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters and results
        best_params = study.best_params
        print(f"Best RMSE: {study.best_value:.4f}")
        print("Best hyperparameters:", best_params)
        
        # Train model with best parameters
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        # Save best parameters to file
        with open(params_file, 'w') as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
    
    # Evaluate the model
    train_results = evaluate_model(best_model, X_train, y_train, "XGBoost - Training")
    val_results = evaluate_model(best_model, X_val, y_val, "XGBoost - Validation")
    
    # Save the model
    joblib.dump(best_model, 'models/xgboost_model.pkl')
    
    return best_model, train_results, val_results

# LightGBM Optimization Function
def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=50, use_previous=False):
    """Optimize LightGBM using Bayesian optimization or load previous parameters"""
    params_file = 'models/lightgbm_params.txt'
    
    # Check if we should use previous parameters
    if use_previous and os.path.exists(params_file):
        print("\n" + "="*50)
        print("Loading previous LightGBM hyperparameters")
        print("="*50)
        
        best_params = load_params_from_file(params_file)
        print("Loaded parameters:", best_params)
        
        # Train model with loaded parameters
        best_model = lgb.LGBMRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                       callbacks=[lgb.early_stopping(50)], eval_metric='rmse')
        
    else:
        print("\n" + "="*50)
        print("Optimizing LightGBM")
        print("="*50)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(50)], eval_metric='rmse')
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters and results
        best_params = study.best_params
        print(f"Best RMSE: {study.best_value:.4f}")
        print("Best hyperparameters:", best_params)
        
        # Train model with best parameters
        best_model = lgb.LGBMRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(50)], eval_metric='rmse')
        
        # Save best parameters to file
        with open(params_file, 'w') as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
    
    # Evaluate the model
    train_results = evaluate_model(best_model, X_train, y_train, "LightGBM - Training")
    val_results = evaluate_model(best_model, X_val, y_val, "LightGBM - Validation")
    
    # Save the model
    joblib.dump(best_model, 'models/lightgbm_model.pkl')
    
    return best_model, train_results, val_results

# CatBoost Optimization Function
def optimize_catboost(X_train, y_train, X_val, y_val, n_trials=50, use_previous=False):
    """Optimize CatBoost using Bayesian optimization or load previous parameters"""
    params_file = 'models/catboost_params.txt'
    
    # Check if we should use previous parameters
    if use_previous and os.path.exists(params_file):
        print("\n" + "="*50)
        print("Loading previous CatBoost hyperparameters")
        print("="*50)
        
        best_params = load_params_from_file(params_file)
        print("Loaded parameters:", best_params)
        
        # Train model with loaded parameters
        best_model = cb.CatBoostRegressor(
            **best_params,
            random_seed=42,
            verbose=100,
            allow_writing_files=False
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        
    else:
        print("\n" + "="*50)
        print("Optimizing CatBoost")
        print("="*50)
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'early_stopping_rounds': 50,
                'random_seed': 42
            }
            
            # Remove early_stopping_rounds from params
            early_stopping_rounds = params.pop('early_stopping_rounds')
            
            model = cb.CatBoostRegressor(
                **params,
                verbose=False,
                allow_writing_files=False
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=early_stopping_rounds, verbose=False)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters and results
        best_params = study.best_params
        print(f"Best RMSE: {study.best_value:.4f}")
        print("Best hyperparameters:", best_params)
        
        # Train model with best parameters
        best_model = cb.CatBoostRegressor(
            **best_params,
            random_seed=42,
            verbose=100,
            allow_writing_files=False
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        
        # Save best parameters to file
        with open(params_file, 'w') as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
    
    # Evaluate the model
    train_results = evaluate_model(best_model, X_train, y_train, "CatBoost - Training")
    val_results = evaluate_model(best_model, X_val, y_val, "CatBoost - Validation")
    
    # Save the model
    best_model.save_model('models/catboost_model.cbm')
    
    return best_model, train_results, val_results

def main():
    # Load the data with top features
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_with_top_features(
        "../../Dataset/processed_Resaleflatprices_XGB.csv", 
        top_features_count=100
    )
    
    # Store all results
    results = {}
    
    # Check what to do with each model
    model_options = {}
    
    # Check for existing parameter files
    xgb_params_exist = os.path.exists('models/xgboost_params.txt')
    lgb_params_exist = os.path.exists('models/lightgbm_params.txt')
    cb_params_exist = os.path.exists('models/catboost_params.txt')
    
    # Check for existing models
    xgb_model_exists = os.path.exists('models/xgboost_model.pkl')
    lgb_model_exists = os.path.exists('models/lightgbm_model.pkl')
    cb_model_exists = os.path.exists('models/catboost_model.cbm')
    
    # Give user options for each model
    if xgb_params_exist or xgb_model_exists:
        print("\nOptions for XGBoost:")
        print("1: Use previous hyperparameters (faster)")
        print("2: Optimize again (slower, might find better parameters)")
        if xgb_model_exists:
            print("3: Load existing model (fastest, no training)")
        choice = input("Select an option [1-3]: ")
        model_options['xgboost'] = int(choice)
    else:
        model_options['xgboost'] = 2  # Optimize if no previous parameters
        
    if lgb_params_exist or lgb_model_exists:
        print("\nOptions for LightGBM:")
        print("1: Use previous hyperparameters (faster)")
        print("2: Optimize again (slower, might find better parameters)")
        if lgb_model_exists:
            print("3: Load existing model (fastest, no training)")
        choice = input("Select an option [1-3]: ")
        model_options['lightgbm'] = int(choice)
    else:
        model_options['lightgbm'] = 2  # Optimize if no previous parameters
        
    if cb_params_exist or cb_model_exists:
        print("\nOptions for CatBoost:")
        print("1: Use previous hyperparameters (faster)")
        print("2: Optimize again (slower, might find better parameters)")
        if cb_model_exists:
            print("3: Load existing model (fastest, no training)")
        choice = input("Select an option [1-3]: ")
        model_options['catboost'] = int(choice)
    else:
        model_options['catboost'] = 2  # Optimize if no previous parameters
    
    # Process XGBoost
    if model_options['xgboost'] == 3 and xgb_model_exists:
        # Load existing model
        print("\nLoading existing XGBoost model...")
        xgb_model = joblib.load('models/xgboost_model.pkl')
        xgb_train_results = evaluate_model(xgb_model, X_train, y_train, "XGBoost - Training")
        xgb_val_results = evaluate_model(xgb_model, X_val, y_val, "XGBoost - Validation")
    else:
        # Either use previous hyperparameters or optimize
        xgb_model, xgb_train_results, xgb_val_results = optimize_xgboost(
            X_train, y_train, X_val, y_val, 
            n_trials=50, 
            use_previous=(model_options['xgboost'] == 1)
        )
    
    # Process LightGBM
    if model_options['lightgbm'] == 3 and lgb_model_exists:
        # Load existing model
        print("\nLoading existing LightGBM model...")
        lgb_model = joblib.load('models/lightgbm_model.pkl')
        lgb_train_results = evaluate_model(lgb_model, X_train, y_train, "LightGBM - Training")
        lgb_val_results = evaluate_model(lgb_model, X_val, y_val, "LightGBM - Validation")
    else:
        # Either use previous hyperparameters or optimize
        lgb_model, lgb_train_results, lgb_val_results = optimize_lightgbm(
            X_train, y_train, X_val, y_val, 
            n_trials=50, 
            use_previous=(model_options['lightgbm'] == 1)
        )
    
    # Process CatBoost
    if model_options['catboost'] == 3 and cb_model_exists:
        # Load existing model
        print("\nLoading existing CatBoost model...")
        cb_model = cb.CatBoostRegressor().load_model('models/catboost_model.cbm')
        cb_train_results = evaluate_model(cb_model, X_train, y_train, "CatBoost - Training")
        cb_val_results = evaluate_model(cb_model, X_val, y_val, "CatBoost - Validation")
    else:
        # Either use previous hyperparameters or optimize
        cb_model, cb_train_results, cb_val_results = optimize_catboost(
            X_train, y_train, X_val, y_val, 
            n_trials=50, 
            use_previous=(model_options['catboost'] == 1)
        )
    
    # Store results
    results = {
        'XGBoost': {
            'train': xgb_train_results,
            'val': xgb_val_results,
            'model': xgb_model
        },
        'LightGBM': {
            'train': lgb_train_results,
            'val': lgb_val_results,
            'model': lgb_model
        },
        'CatBoost': {
            'train': cb_train_results,
            'val': cb_val_results,
            'model': cb_model
        }
    }
    
    # Save results dictionary
    joblib.dump(results, 'results/algorithm_results.pkl')
    
    # Plot validation predictions for each algorithm
    for alg_name, alg_results in results.items():
        plot_predictions(
            y_val, 
            alg_results['val']['predictions'],
            title=f"{alg_name} Validation Predictions",
            filename=f"{alg_name.lower()}_val_predictions"
        )
    
    # Compare algorithms
    print("\nComparing algorithms on validation set...")
    
    # Compare by RMSE
    best_rmse_alg, _ = compare_algorithms(results, metric='rmse', dataset='val')
    
    # Compare by R²
    best_r2_alg, _ = compare_algorithms(results, metric='r2', dataset='val')
    
    # Compare by MAPE
    best_mape_alg, _ = compare_algorithms(results, metric='mape', dataset='val')
    
    # Finally, evaluate on test data using the best model by RMSE
    best_model = results[best_rmse_alg]['model']
    print(f"\nEvaluating best model ({best_rmse_alg}) on test data...")
    test_results = evaluate_model(best_model, X_test, y_test, f"{best_rmse_alg} - Test")
    
    # Plot test predictions for best model
    plot_predictions(
        y_test,
        test_results['predictions'],
        title=f"{best_rmse_alg} Test Predictions",
        filename=f"best_model_test_predictions"
    )
    
    print("\nMulti-Algorithm Optimization completed!")
    print(f"The best model by RMSE is: {best_rmse_alg}")
    print(f"The best model by R² is: {best_r2_alg}")
    print(f"The best model by MAPE is: {best_mape_alg}")
    
    # Save test results
    with open('results/test_results.txt', 'w') as f:
        f.write(f"Best model by RMSE: {best_rmse_alg}\n")
        f.write(f"RMSE: ${test_results['rmse']:.2f}\n")
        f.write(f"R² Score: {test_results['r2']:.4f}\n")
        f.write(f"MAE: ${test_results['mae']:.2f}\n")
        f.write(f"MAPE: {test_results['mape']:.2f}%\n")

if __name__ == "__main__":
    main()
