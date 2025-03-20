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
from scipy.optimize import minimize
import lightgbm as lgb
from tqdm import tqdm

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for models and results
os.makedirs('models/ensemble', exist_ok=True)
os.makedirs('results/ensemble', exist_ok=True)

def load_data_with_top_features(data_path, top_features_count=100):
    """
    Load data and filter to only use top features - modified to match K-fold script
    """
    print(f"Loading data from {data_path}...")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Split into train/test first to maintain consistency with previous steps
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Get target column
    target_col = 'resale_price'
    
    # Check if feature importance file exists
    feature_file = 'feature_analysis/xgboost_importance.csv'  # Use same file as K-fold script
    
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
    
    # Split for evaluation - we'll use the same split as K-fold did for initial validation
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)
    
    # Create feature sets
    X_train = train[top_features]
    y_train = train[target_col]
    
    X_val = val[top_features]
    y_val = val[target_col]
    
    X_test = test[top_features]
    y_test = test[target_col]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, top_features

def evaluate_prediction(y_true, y_pred, dataset_name=""):
    """Evaluate prediction performance"""
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Print metrics
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'predictions': y_pred
    }

def load_kfold_models(n_folds=5):
    """Load the trained K-fold LightGBM models"""
    models = []
    for i in range(1, n_folds + 1):
        model_path = f'models/kfold/lightgbm_fold{i}.pkl'
        if os.path.exists(model_path):
            models.append(joblib.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Could not find model at {model_path}")
            return None
    return models

def generate_model_predictions(models, X_train, X_val, X_test):
    """Generate predictions from all models on train, val, and test sets"""
    print("Generating predictions from all models...")
    
    train_preds = []
    val_preds = []
    test_preds = []
    
    for i, model in enumerate(models):
        print(f"Generating predictions for Model {i+1}")
        train_preds.append(model.predict(X_train))
        val_preds.append(model.predict(X_val))
        test_preds.append(model.predict(X_test))
    
    # Convert to numpy arrays
    train_preds = np.array(train_preds).T  # shape: (n_samples, n_models)
    val_preds = np.array(val_preds).T      # shape: (n_samples, n_models)
    test_preds = np.array(test_preds).T    # shape: (n_samples, n_models)
    
    return train_preds, val_preds, test_preds

def optimize_weights_grid_search(val_preds, y_val, n_points=10):
    """
    Find optimal weights using grid search for 5 models.
    This is a simpler approach than constrained optimization.
    """
    print("Optimizing weights using grid search...")
    
    n_models = val_preds.shape[1]
    
    if n_models > 5:
        print("Warning: Grid search may be slow for more than 5 models.")
    
    # Create a grid of weights that sum to 1
    best_rmse = float('inf')
    best_weights = None
    
    # For 5 models, we can do a 4D grid search (the 5th weight is determined by the constraint)
    # With n_points=10 per dimension, that's 10^4 = 10,000 combinations
    
    total_combinations = n_points**(n_models-1)
    print(f"Testing {total_combinations} weight combinations...")
    
    # Create weight grid
    grid_points = np.linspace(0, 1, n_points)
    
    # Progress tracking
    counter = 0
    pbar = tqdm(total=total_combinations)
    
    # For 5 models
    if n_models == 5:
        for w1 in grid_points:
            for w2 in grid_points:
                for w3 in grid_points:
                    for w4 in grid_points:
                        # Calculate w5 to ensure sum = 1
                        w5 = 1 - (w1 + w2 + w3 + w4)
                        
                        # Skip invalid combinations
                        if w5 < 0:
                            pbar.update(1)
                            counter += 1
                            continue
                        
                        weights = np.array([w1, w2, w3, w4, w5])
                        weighted_pred = np.dot(val_preds, weights)
                        rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_weights = weights.copy()
                        
                        pbar.update(1)
                        counter += 1
    
    # For 4 models
    elif n_models == 4:
        for w1 in grid_points:
            for w2 in grid_points:
                for w3 in grid_points:
                    # Calculate w4 to ensure sum = 1
                    w4 = 1 - (w1 + w2 + w3)
                    
                    # Skip invalid combinations
                    if w4 < 0:
                        pbar.update(1)
                        counter += 1
                        continue
                    
                    weights = np.array([w1, w2, w3, w4])
                    weighted_pred = np.dot(val_preds, weights)
                    rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = weights.copy()
                    
                    pbar.update(1)
                    counter += 1
    
    # For 3 models
    elif n_models == 3:
        for w1 in grid_points:
            for w2 in grid_points:
                # Calculate w3 to ensure sum = 1
                w3 = 1 - (w1 + w2)
                
                # Skip invalid combinations
                if w3 < 0:
                    pbar.update(1)
                    counter += 1
                    continue
                
                weights = np.array([w1, w2, w3])
                weighted_pred = np.dot(val_preds, weights)
                rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights.copy()
                
                pbar.update(1)
                counter += 1
    else:
        # For any number of models, but will be slow for large n_models
        # Use recursive function or another approach for more than 5 models
        print("Falling back to optimization algorithm for more than 5 models...")
        return optimize_weights(val_preds, y_val)
    
    pbar.close()
    print(f"Tested {counter} valid combinations")
    print(f"Best weights: {best_weights}")
    print(f"Best RMSE: ${best_rmse:.2f}")
    
    return best_weights

def optimize_weights(val_preds, y_val):
    """Find optimal weights using constrained optimization"""
    print("Optimizing weights using constrained optimization...")
    
    n_models = val_preds.shape[1]
    
    # Objective function: RMSE of weighted prediction
    def objective(weights):
        # Reshape weights to 1D array for dot product
        weights_1d = weights.reshape(-1)
        weighted_pred = np.dot(val_preds, weights_1d)
        rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
        return rmse
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': True, 'maxiter': 1000}
    )
    
    optimal_weights = result.x
    print(f"Optimization successful: {result.success}")
    print(f"Optimal weights: {optimal_weights}")
    print(f"Optimized RMSE: ${result.fun:.2f}")
    
    return optimal_weights

def train_weighted_ensemble(train_preds, val_preds, test_preds, y_train, y_val, y_test):
    """Train and evaluate a weighted ensemble model"""
    print("\n" + "="*50)
    print("Building Weighted Ensemble Model")
    print("="*50)
    
    # First, evaluate uniform (simple average) ensemble
    uniform_weights = np.ones(train_preds.shape[1]) / train_preds.shape[1]
    
    uniform_train_pred = np.dot(train_preds, uniform_weights)
    uniform_val_pred = np.dot(val_preds, uniform_weights)
    uniform_test_pred = np.dot(test_preds, uniform_weights)
    
    print("\nUniform Weights Ensemble Evaluation:")
    uniform_train_results = evaluate_prediction(y_train, uniform_train_pred, "Training (Uniform Weights)")
    uniform_val_results = evaluate_prediction(y_val, uniform_val_pred, "Validation (Uniform Weights)")
    uniform_test_results = evaluate_prediction(y_test, uniform_test_pred, "Test (Uniform Weights)")
    
    # Now find optimal weights
    # Using grid search for 5 models
    if val_preds.shape[1] <= 5:
        optimal_weights = optimize_weights_grid_search(val_preds, y_val)
    else:
        # For more than 5 models, use scipy's optimize
        optimal_weights = optimize_weights(val_preds, y_val)
    
    # Evaluate weighted ensemble
    weighted_train_pred = np.dot(train_preds, optimal_weights)
    weighted_val_pred = np.dot(val_preds, optimal_weights)
    weighted_test_pred = np.dot(test_preds, optimal_weights)
    
    print("\nOptimal Weights Ensemble Evaluation:")
    weighted_train_results = evaluate_prediction(y_train, weighted_train_pred, "Training (Optimal Weights)")
    weighted_val_results = evaluate_prediction(y_val, weighted_val_pred, "Validation (Optimal Weights)")
    weighted_test_results = evaluate_prediction(y_test, weighted_test_pred, "Test (Optimal Weights)")
    
    # Compare uniform vs weighted ensemble
    print("\nComparison: Optimal Weights vs Uniform Weights (Test Set)")
    metrics = ['rmse', 'r2', 'mae', 'mape']
    
    for metric in metrics:
        if metric == 'r2':
            improvement = ((weighted_test_results[metric] / uniform_test_results[metric]) - 1) * 100
            better = weighted_test_results[metric] > uniform_test_results[metric]
        else:
            improvement = ((uniform_test_results[metric] / weighted_test_results[metric]) - 1) * 100
            better = weighted_test_results[metric] < uniform_test_results[metric]
            
        sign = "+" if improvement > 0 else ""
        print(f"{metric.upper()}: {'Improved' if better else 'Declined'} by {sign}{improvement:.2f}%")
    
    # Save the ensemble model (the weights)
    ensemble_model = {
        'uniform_weights': uniform_weights,
        'optimal_weights': optimal_weights,
        'n_models': train_preds.shape[1]
    }
    
    joblib.dump(ensemble_model, 'models/ensemble/weighted_ensemble.pkl')
    print("\nEnsemble model saved to 'models/ensemble/weighted_ensemble.pkl'")
    
    # Plot weights distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(optimal_weights) + 1), optimal_weights, color='skyblue')
    plt.axhline(y=1/len(optimal_weights), color='r', linestyle='--', label=f'Uniform Weight: {1/len(optimal_weights):.3f}')
    
    for i, w in enumerate(optimal_weights):
        plt.text(i + 1, w + 0.01, f"{w:.3f}", ha='center')
        
    plt.xlabel('Model')
    plt.ylabel('Weight')
    plt.title('Optimal Weights for Ensemble Model')
    plt.xticks(range(1, len(optimal_weights) + 1))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/ensemble/model_weights.png')
    plt.close()
    
    # Plot test predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, weighted_test_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_test.max(), weighted_test_pred.max())
    min_val = min(y_test.min(), weighted_test_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Weighted Ensemble Predicted Price')
    plt.title('Weighted Ensemble: Actual vs Predicted Prices on Test Set')
    plt.tight_layout()
    plt.savefig('results/ensemble/weighted_ensemble_test_predictions.png')
    plt.close()
    
    # Save test results
    weighted_test_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': weighted_test_pred,
        'Error': y_test - weighted_test_pred,
        'Absolute_Error': np.abs(y_test - weighted_test_pred),
        'Percentage_Error': np.abs((y_test - weighted_test_pred) / y_test) * 100
    })
    weighted_test_df.to_csv('results/ensemble/weighted_ensemble_test_predictions.csv', index=False)
    
    # Save detailed results
    with open('results/ensemble/weighted_ensemble_results.txt', 'w') as f:
        f.write("Weighted Ensemble Model Results\n")
        f.write("="*50 + "\n")
        f.write("Optimal Weights: " + str([f"{w:.4f}" for w in optimal_weights]) + "\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"RMSE: ${weighted_test_results['rmse']:.2f}\n")
        f.write(f"R² Score: {weighted_test_results['r2']:.4f}\n")
        f.write(f"MAE: ${weighted_test_results['mae']:.2f}\n")
        f.write(f"MAPE: {weighted_test_results['mape']:.2f}%\n\n")
        
        f.write("Comparison with Uniform Weights Ensemble:\n")
        for metric in metrics:
            if metric == 'r2':
                improvement = ((weighted_test_results[metric] / uniform_test_results[metric]) - 1) * 100
                better = weighted_test_results[metric] > uniform_test_results[metric]
            else:
                improvement = ((uniform_test_results[metric] / weighted_test_results[metric]) - 1) * 100
                better = weighted_test_results[metric] < uniform_test_results[metric]
                
            sign = "+" if improvement > 0 else ""
            f.write(f"{metric.upper()}: {'Improved' if better else 'Declined'} by {sign}{improvement:.2f}%\n")
    
    return {
        'uniform': {
            'weights': uniform_weights,
            'train_results': uniform_train_results,
            'val_results': uniform_val_results,
            'test_results': uniform_test_results
        },
        'weighted': {
            'weights': optimal_weights,
            'train_results': weighted_train_results,
            'val_results': weighted_val_results,
            'test_results': weighted_test_results
        }
    }

def evaluate_ensemble_with_individual_models(models, X_test, y_test, weighted_ensemble):
    """Compare the weighted ensemble with individual models on test set"""
    print("\n" + "="*50)
    print("Comparing Weighted Ensemble with Individual Models")
    print("="*50)
    
    # Get ensemble predictions and results
    weighted_test_pred = weighted_ensemble['weighted']['test_results']['predictions']
    weighted_test_results = weighted_ensemble['weighted']['test_results']
    
    # Get individual model predictions and results
    individual_results = []
    
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        results = evaluate_prediction(y_test, y_pred, f"Model {i+1} on Test")
        individual_results.append(results)
    
    # Find the best individual model for each metric
    metrics = ['rmse', 'r2', 'mae', 'mape']
    comparison = {}
    
    for metric in metrics:
        # Get individual metrics
        individual_metrics = [result[metric] for result in individual_results]
        
        # For R², higher is better; for others, lower is better
        if metric == 'r2':
            best_individual_idx = np.argmax(individual_metrics)
            is_ensemble_better = weighted_test_results[metric] > individual_metrics[best_individual_idx]
        else:
            best_individual_idx = np.argmin(individual_metrics)
            is_ensemble_better = weighted_test_results[metric] < individual_metrics[best_individual_idx]
        
        best_individual = individual_metrics[best_individual_idx]
        
        # Calculate improvement percentage
        if metric == 'r2':
            improvement = ((weighted_test_results[metric] / best_individual) - 1) * 100
        else:
            improvement = ((best_individual / weighted_test_results[metric]) - 1) * 100
        
        comparison[metric] = {
            'best_individual': best_individual,
            'best_individual_idx': best_individual_idx,
            'ensemble': weighted_test_results[metric],
            'improvement': improvement,
            'is_ensemble_better': is_ensemble_better
        }
    
    # Print comparison
    print("\nWeighted Ensemble vs Best Individual Model Comparison:")
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
        labels = [f"Model {j+1}" for j in range(len(models))] + ["Weighted\nEnsemble"]
        
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
    plt.savefig('results/ensemble/weighted_ensemble_vs_individual.png')
    plt.close()
    
    # Save detailed comparison
    with open('results/ensemble/weighted_ensemble_vs_individual.txt', 'w') as f:
        f.write("Weighted Ensemble vs Individual Models Comparison\n")
        f.write("="*50 + "\n")
        
        for metric, data in comparison.items():
            sign = "+" if data['improvement'] > 0 else ""
            f.write(f"{metric.upper()}: Ensemble = {data['ensemble']:.4f}, " +
                    f"Best Individual = {data['best_individual']:.4f} (Model {data['best_individual_idx']+1}), " +
                    f"Improvement: {sign}{data['improvement']:.2f}%\n")
    
    return comparison

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data_with_top_features(
        "../../Dataset/processed_Resaleflatprices_XGB.csv", 
        top_features_count=100
    )
    
    # Load k-fold models
    n_folds = 5
    models = load_kfold_models(n_folds)
    
    if models is None or len(models) != n_folds:
        print(f"Error: Could not load all {n_folds} K-fold models. Please train them first using LightGBM_kfold.py")
        return
    
    # Generate predictions from all models
    train_preds, val_preds, test_preds = generate_model_predictions(models, X_train, X_val, X_test)
    
    # Train weighted ensemble
    ensemble_results = train_weighted_ensemble(train_preds, val_preds, test_preds, y_train, y_val, y_test)
    
    # Compare weighted ensemble with individual models
    comparison = evaluate_ensemble_with_individual_models(models, X_test, y_test, ensemble_results)
    
    print("\nWeighted Ensemble Building completed!")
    print("Results saved in 'results/ensemble/' directory")

if __name__ == "__main__":
    main()
