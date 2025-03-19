import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
from tqdm import tqdm

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for models and results
os.makedirs('models/kfold', exist_ok=True)
os.makedirs('results/kfold', exist_ok=True)

def load_data_with_top_features(data_path, top_features_count=100):
    """
    Load data and filter to only use top features
    """
    print(f"Loading data from {data_path}...")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Split into train/test first to maintain consistency with previous steps
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    
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
    
    # For K-fold cross-validation, we'll use the train_val set
    X = train_val[top_features]
    y = train_val[target_col]
    
    # Still keep the test set separate for final evaluation
    X_test = test[top_features]
    y_test = test[target_col]
    
    print(f"Training/Validation data: {X.shape}")
    print(f"Test data: {X_test.shape}")
    
    return X, y, X_test, y_test, top_features

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

def plot_kfold_metrics(fold_results, metric='rmse'):
    """Plot metrics across K folds"""
    plt.figure(figsize=(10, 6))
    
    # Extract metrics for each fold
    folds = list(range(1, len(fold_results) + 1))
    metrics = [result[metric] for result in fold_results]
    
    # Calculate mean and std
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    
    # Create the bar plot
    plt.bar(folds, metrics, color='skyblue', alpha=0.7)
    plt.axhline(y=mean_metric, color='r', linestyle='-', label=f'Mean: {mean_metric:.2f}')
    plt.axhline(y=mean_metric + std_metric, color='g', linestyle='--', alpha=0.7, 
                label=f'±1 Std: {std_metric:.2f}')
    plt.axhline(y=mean_metric - std_metric, color='g', linestyle='--', alpha=0.7)
    
    # Format plot
    title_map = {
        'rmse': 'RMSE across K Folds',
        'r2': 'R² Score across K Folds',
        'mae': 'MAE across K Folds',
        'mape': 'MAPE (%) across K Folds'
    }
    
    plt.title(title_map.get(metric, f'{metric.upper()} across K Folds'))
    plt.xlabel('Fold Number')
    
    y_label_map = {
        'rmse': 'RMSE ($)',
        'r2': 'R² Score',
        'mae': 'MAE ($)',
        'mape': 'MAPE (%)'
    }
    plt.ylabel(y_label_map.get(metric, metric.upper()))
    
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'results/kfold/lightgbm_kfold_{metric}.png')
    plt.close()
    
    return mean_metric, std_metric

def plot_feature_importance(model, feature_names, fold=None):
    """Plot feature importance from the model"""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    feat_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feat_importances = feat_importances.sort_values('Importance', ascending=False)
    
    # Save to CSV for future use
    folder_path = 'feature_analysis/kfold'
    os.makedirs(folder_path, exist_ok=True)
    
    filename = 'lightgbm_importance.csv'
    if fold is not None:
        filename = f'lightgbm_fold{fold}_importance.csv'
        
    feat_importances.to_csv(f'{folder_path}/{filename}', index=False)
    
    # Display top 20 features
    top_features = feat_importances.head(20)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    title = 'Top 20 Feature Importances (LightGBM)'
    if fold is not None:
        title = f'Top 20 Feature Importances - Fold {fold} (LightGBM)'
        
    plt.title(title)
    plt.tight_layout()
    
    filename = 'lightgbm_feature_importance.png'
    if fold is not None:
        filename = f'lightgbm_fold{fold}_feature_importance.png'
        
    plt.savefig(f'results/kfold/{filename}')
    plt.close()
    
    return feat_importances

def load_best_params():
    """Load best LightGBM parameters from previous optimization"""
    params_file = 'models/lightgbm_params.txt'
    
    if os.path.exists(params_file):
        params = {}
        with open(params_file, 'r') as f:
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
    else:
        print("No previously optimized parameters found. Using default parameters.")
        return {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }

def train_kfold_models(X, y, n_folds=5, random_state=42):
    """Train multiple LightGBM models using K-fold cross-validation"""
    print(f"\n{'='*50}")
    print(f"Training {n_folds} LightGBM Models with K-Fold Cross-Validation")
    print(f"{'='*50}")
    
    # Load best parameters from previous optimization (Step 4)
    best_params = load_best_params()
    print("Using parameters:", best_params)
    
    # Initialize K-fold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Store models and results
    models = []
    fold_results = []
    feature_importances = []
    
    # Train models for each fold
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=n_folds, desc="Training folds")):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model with best parameters
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)],
            eval_metric='rmse'
        )
        
        # Evaluate on validation set
        val_results = evaluate_model(model, X_val, y_val, f"Fold {fold+1} Validation")
        fold_results.append(val_results)
        
        # Get feature importance
        fold_importance = plot_feature_importance(model, X.columns, fold=fold+1)
        feature_importances.append(fold_importance)
        
        # Save the model
        joblib.dump(model, f'models/kfold/lightgbm_fold{fold+1}.pkl')
        models.append(model)
        
        print(f"Fold {fold+1} model saved.")
    
    # Calculate and plot aggregate metrics across folds
    print("\nK-Fold Cross-Validation Results:")
    avg_metrics = {}
    
    for metric in ['rmse', 'r2', 'mae', 'mape']:
        mean_metric, std_metric = plot_kfold_metrics(fold_results, metric)
        avg_metrics[metric] = {'mean': mean_metric, 'std': std_metric}
        print(f"Average {metric.upper()}: {mean_metric:.4f} ± {std_metric:.4f}")
    
    # Create aggregate feature importance
    all_importances = pd.concat(feature_importances).groupby('Feature').mean().reset_index()
    all_importances = all_importances.sort_values('Importance', ascending=False)
    
    # Save aggregate feature importance
    all_importances.to_csv('feature_analysis/kfold/lightgbm_aggregate_importance.csv', index=False)
    
    # Plot aggregate feature importance (top 20)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=all_importances.head(20))
    plt.title('Top 20 Aggregate Feature Importances (LightGBM)')
    plt.tight_layout()
    plt.savefig('results/kfold/lightgbm_aggregate_feature_importance.png')
    plt.close()
    
    # Save results summary
    with open('results/kfold/lightgbm_kfold_summary.txt', 'w') as f:
        f.write(f"K-Fold Cross-Validation with {n_folds} folds\n")
        f.write("="*50 + "\n")
        for metric, values in avg_metrics.items():
            f.write(f"Average {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}\n")
    
    return models, fold_results, avg_metrics, all_importances

def evaluate_models_on_test_set(models, X_test, y_test):
    """Evaluate all K-fold models on the test set"""
    print(f"\n{'='*50}")
    print("Evaluating K-Fold Models on Test Set")
    print(f"{'='*50}")
    
    # Individual model predictions
    print("\nIndividual Model Evaluations:")
    individual_results = []
    all_predictions = []
    
    for i, model in enumerate(models):
        results = evaluate_model(model, X_test, y_test, f"Model {i+1} on Test")
        individual_results.append(results)
        all_predictions.append(results['predictions'])
    
    # Create an ensemble prediction (simple average)
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Evaluate ensemble
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    
    print("\nEnsemble Model (Simple Average) Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    ensemble_results = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'predictions': ensemble_pred
    }
    
    # Compare individual models with ensemble
    metrics = ['rmse', 'r2', 'mae', 'mape']
    comparison = {}
    
    for metric in metrics:
        # Get individual metrics
        individual_metrics = [result[metric] for result in individual_results]
        
        # For R², higher is better; for others, lower is better
        if metric == 'r2':
            best_individual_idx = np.argmax(individual_metrics)
            is_ensemble_better = ensemble_results[metric] > individual_metrics[best_individual_idx]
        else:
            best_individual_idx = np.argmin(individual_metrics)
            is_ensemble_better = ensemble_results[metric] < individual_metrics[best_individual_idx]
        
        best_individual = individual_metrics[best_individual_idx]
        
        # Calculate improvement percentage
        if metric == 'r2':
            improvement = ((ensemble_results[metric] / best_individual) - 1) * 100
        else:
            improvement = ((best_individual / ensemble_results[metric]) - 1) * 100
        
        comparison[metric] = {
            'best_individual': best_individual,
            'best_individual_idx': best_individual_idx,
            'ensemble': ensemble_results[metric],
            'improvement': improvement,
            'is_ensemble_better': is_ensemble_better
        }
    
    # Print comparison
    print("\nEnsemble vs Best Individual Model Comparison:")
    for metric, data in comparison.items():
        sign = "+" if data['improvement'] > 0 else ""
        print(f"{metric.upper()}: Ensemble = {data['ensemble']:.4f}, " +
              f"Best Individual = {data['best_individual']:.4f} (Model {data['best_individual_idx']+1}), " +
              f"Improvement: {sign}{data['improvement']:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    
    # Create subplot for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        data = comparison[metric]
        
        # Individual model metrics
        individual_metrics = [result[metric] for result in individual_results]
        
        # Add ensemble as last bar
        all_metrics = individual_metrics + [data['ensemble']]
        labels = [f"Model {j+1}" for j in range(len(models))] + ["Ensemble"]
        
        # Determine color (highlight best)
        best_idx = len(all_metrics) - 1 if data['is_ensemble_better'] else data['best_individual_idx']
        colors = ['#ff9999' if j != best_idx else '#66b3ff' for j in range(len(all_metrics))]
        
        plt.bar(labels, all_metrics, color=colors)
        
        # Add value labels
        for j, v in enumerate(all_metrics):
            if metric == 'mape':
                plt.text(j, v + 0.1, f"{v:.2f}%", ha='center')
            elif metric == 'r2':
                plt.text(j, v + 0.005, f"{v:.4f}", ha='center')
            else:
                plt.text(j, v + (max(all_metrics) * 0.01), f"${v:.2f}", ha='center')
        
        plt.title(f"{metric.upper()} Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/kfold/model_ensemble_comparison.png')
    plt.close()
    
    # Plot test predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, ensemble_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_test.max(), ensemble_pred.max())
    min_val = min(y_test.min(), ensemble_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Ensemble Predicted Price')
    plt.title('Ensemble Model: Actual vs Predicted Prices on Test Set')
    plt.tight_layout()
    plt.savefig('results/kfold/ensemble_test_predictions.png')
    plt.close()
    
    # Save ensemble predictions
    ensemble_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': ensemble_pred,
        'Error': y_test - ensemble_pred,
        'Absolute_Error': np.abs(y_test - ensemble_pred),
        'Percentage_Error': np.abs((y_test - ensemble_pred) / y_test) * 100
    })
    ensemble_df.to_csv('results/kfold/ensemble_test_predictions.csv', index=False)
    
    # Save detailed results
    with open('results/kfold/ensemble_test_results.txt', 'w') as f:
        f.write("Ensemble Model (Simple Average) Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"RMSE: ${rmse:.2f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"MAE: ${mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n\n")
        
        f.write("Comparison with Individual Models\n")
        f.write("-"*50 + "\n")
        for metric, data in comparison.items():
            sign = "+" if data['improvement'] > 0 else ""
            f.write(f"{metric.upper()}: Ensemble = {data['ensemble']:.4f}, " +
                    f"Best Individual = {data['best_individual']:.4f} (Model {data['best_individual_idx']+1}), " +
                    f"Improvement: {sign}{data['improvement']:.2f}%\n")
    
    return individual_results, ensemble_results, comparison

def main():
    # Set the number of folds
    n_folds = 5
    
    # Load the data with top features
    X, y, X_test, y_test, feature_names = load_data_with_top_features(
        "../../Dataset/processed_Resaleflatprices_XGB.csv", 
        top_features_count=100
    )
    
    # Check if k-fold models already exist
    all_models_exist = True
    for i in range(1, n_folds + 1):
        if not os.path.exists(f'models/kfold/lightgbm_fold{i}.pkl'):
            all_models_exist = False
            break
    
    # Ask whether to retrain or use existing models
    if all_models_exist:
        retrain = input("\nK-fold models already exist. Retrain them? (y/n): ").lower() == 'y'
    else:
        retrain = True
    
    # Either train new models or load existing ones
    if retrain:
        # Train k-fold models
        models, fold_results, avg_metrics, all_importances = train_kfold_models(X, y, n_folds=n_folds)
    else:
        print("\nLoading existing k-fold models...")
        models = []
        for i in range(1, n_folds + 1):
            model_path = f'models/kfold/lightgbm_fold{i}.pkl'
            if os.path.exists(model_path):
                models.append(joblib.load(model_path))
                print(f"Loaded model from {model_path}")
            else:
                print(f"Warning: Could not find model at {model_path}")
        
        # Load aggregate feature importance if available
        importance_path = 'feature_analysis/kfold/lightgbm_aggregate_importance.csv'
        if os.path.exists(importance_path):
            all_importances = pd.read_csv(importance_path)
        else:
            all_importances = None
    
    # Evaluate models on test set
    individual_results, ensemble_results, comparison = evaluate_models_on_test_set(models, X_test, y_test)
    
    print("\nK-Fold Model Training and Evaluation completed!")
    print(f"Results saved in 'results/kfold/' directory")

if __name__ == "__main__":
    main()
