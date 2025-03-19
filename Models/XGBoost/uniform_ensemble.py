import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Make sure this is imported at the top level
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, median_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
import lightgbm as lgb

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for models and results
os.makedirs('models/ensemble', exist_ok=True)
os.makedirs('results/ensemble', exist_ok=True)

def load_data_with_top_features(data_path, top_features_count=100):
    """
    Load data and filter to only use top features - consistent with K-fold script
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
    
    # Split for evaluation
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
    medae = median_absolute_error(y_true, y_pred)  # Added Median Absolute Error
    max_err = max_error(y_true, y_pred)  # Added Maximum Error
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate 95% prediction interval
    residuals = y_true - y_pred
    interval_width = np.percentile(np.abs(residuals), 95) * 2
    coverage = np.mean(np.abs(residuals) <= interval_width/2)
    
    # Create a results dictionary
    results = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'medae': medae,
        'max_error': max_err,
        'interval_95': interval_width,
        'coverage': coverage,
        'predictions': y_pred
    }
    
    # Print metrics if dataset_name is provided
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Median AE: ${medae:.2f}")
        print(f"Maximum Error: ${max_err:.2f}")
        print(f"95% Interval Width: ${interval_width:.2f}")
    
    return results

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

def calculate_additional_metrics(models, X, y, feature_names):
    """
    Calculate Adjusted R², CV R² Mean, and CV RMSE Mean for ensemble and individual models
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    X : DataFrame
        Feature data
    y : Series
        Target values
    feature_names : list
        Names of features in the model
        
    Returns:
    --------
    ensemble_metrics : dict
        Dictionary with additional metrics for the ensemble
    individual_metrics : list
        List of dictionaries with additional metrics for each individual model
    """
    # Number of samples and features
    n = X.shape[0]
    p = len(feature_names)
    
    # Configure cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    individual_metrics = []
    
    print("\n" + "="*50)
    print("CALCULATING ADDITIONAL METRICS")
    print("="*50)
    
    # 1. First calculate metrics for individual models
    for i, model in enumerate(models):
        # Regular prediction
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Adjusted R2
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        # Cross-validation R2 scores
        cv_r2_scores = []
        cv_rmse_scores = []
        
        # Measure prediction time
        n_repeat = 10  # Repeat for more accurate timing
        prediction_times = []
        
        for _ in range(n_repeat):
            # Measure single prediction time (on a subset for efficiency)
            sample_size = min(1000, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            
            start_time = time.time()
            model.predict(X_sample)
            pred_time = (time.time() - start_time) / sample_size  # Per sample prediction time
            prediction_times.append(pred_time)
            
        # Average prediction time in milliseconds per sample
        avg_prediction_time = np.mean(prediction_times) * 1000  # Convert to ms
        
        # Manual cross-validation to avoid retraining models
        for train_idx, test_idx in cv.split(X):
            # Get X and y for this fold
            X_cv_test = X.iloc[test_idx]
            y_cv_test = y.iloc[test_idx]
            
            # Predict and evaluate
            y_cv_pred = model.predict(X_cv_test)
            cv_r2 = r2_score(y_cv_test, y_cv_pred)
            cv_rmse = np.sqrt(mean_squared_error(y_cv_test, y_cv_pred))
            
            cv_r2_scores.append(cv_r2)
            cv_rmse_scores.append(cv_rmse)
        
        # Calculate means and stds
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_r2_std = np.std(cv_r2_scores)
        cv_rmse_mean = np.mean(cv_rmse_scores)
        cv_rmse_std = np.std(cv_rmse_scores)
        
        # Print results
        print(f"Model {i+1} Additional Metrics:")
        print(f"  Adjusted R²: {adj_r2:.4f}")
        print(f"  CV R² Mean: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
        print(f"  CV RMSE Mean: ${cv_rmse_mean:.2f} (±${cv_rmse_std:.2f})")
        print(f"  Prediction Time: {avg_prediction_time:.4f} ms/sample")
        print("-" * 40)
        
        # Store metrics
        individual_metrics.append({
            'model_idx': i,
            'adj_r2': adj_r2,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std,
            'prediction_time_ms': avg_prediction_time
        })
    
    # 2. Calculate metrics for ensemble
    # Custom function for ensemble prediction
    def ensemble_predict(X_data):
        ensemble_preds = np.zeros(len(X_data))
        for model in models:
            ensemble_preds += model.predict(X_data)
        return ensemble_preds / len(models)
    
    # Regular prediction
    y_pred_ensemble = ensemble_predict(X)
    r2_ensemble = r2_score(y, y_pred_ensemble)
    
    # Measure ensemble prediction time
    n_repeat = 10  # Repeat for more accurate timing
    prediction_times_ens = []
    
    for _ in range(n_repeat):
        # Measure single prediction time (on a subset for efficiency)
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        
        start_time = time.time()
        ensemble_predict(X_sample)
        pred_time = (time.time() - start_time) / sample_size  # Per sample prediction time
        prediction_times_ens.append(pred_time)
        
    # Average prediction time in milliseconds per sample
    avg_prediction_time_ens = np.mean(prediction_times_ens) * 1000  # Convert to ms
    
    # Adjusted R2
    adj_r2_ensemble = 1 - ((1 - r2_ensemble) * (n - 1) / (n - p - 1))
    
    # Cross-validation metrics for ensemble
    cv_r2_scores_ensemble = []
    cv_rmse_scores_ensemble = []
    
    for train_idx, test_idx in cv.split(X):
        # Get X and y for this fold
        X_cv_test = X.iloc[test_idx]
        y_cv_test = y.iloc[test_idx]
        
        # Predict with ensemble
        y_cv_pred_ensemble = ensemble_predict(X_cv_test)
        
        # Calculate metrics
        cv_r2_ensemble = r2_score(y_cv_test, y_cv_pred_ensemble)
        cv_rmse_ensemble = np.sqrt(mean_squared_error(y_cv_test, y_cv_pred_ensemble))
        
        cv_r2_scores_ensemble.append(cv_r2_ensemble)
        cv_rmse_scores_ensemble.append(cv_rmse_ensemble)
    
    # Calculate means and stds
    cv_r2_mean_ensemble = np.mean(cv_r2_scores_ensemble)
    cv_r2_std_ensemble = np.std(cv_r2_scores_ensemble)
    cv_rmse_mean_ensemble = np.mean(cv_rmse_scores_ensemble)
    cv_rmse_std_ensemble = np.std(cv_rmse_scores_ensemble)
    
    # Print results
    print("\nUniform Ensemble Additional Metrics:")
    print(f"  Adjusted R²: {adj_r2_ensemble:.4f}")
    print(f"  CV R² Mean: {cv_r2_mean_ensemble:.4f} (±{cv_r2_std_ensemble:.4f})")
    print(f"  CV RMSE Mean: ${cv_rmse_mean_ensemble:.2f} (±${cv_rmse_std_ensemble:.2f})")
    print(f"  Prediction Time: {avg_prediction_time_ens:.4f} ms/sample")
    print("-" * 40)
    
    # Store and return results
    ensemble_metrics = {
        'adj_r2': adj_r2_ensemble,
        'cv_r2_mean': cv_r2_mean_ensemble,
        'cv_r2_std': cv_r2_std_ensemble,
        'cv_rmse_mean': cv_rmse_mean_ensemble,
        'cv_rmse_std': cv_rmse_std_ensemble,
        'prediction_time_ms': avg_prediction_time_ens
    }
    
    return ensemble_metrics, individual_metrics

def build_uniform_ensemble(models, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Build and evaluate a uniform ensemble model"""
    print("\n" + "="*50)
    print("Building Uniform Ensemble Model")
    print("="*50)
    
    # Measure total training time for the ensemble approach
    # This includes loading models and all evaluation steps
    import time  # Import time at the function level to prevent shadowing
    training_start_time = time.time()
    
    # Generate individual model predictions
    print("Generating predictions from all models...")
    n_models = len(models)
    
    # For training data
    train_preds = np.zeros((len(y_train), n_models))
    for i, model in enumerate(models):
        train_preds[:, i] = model.predict(X_train)
    
    # For validation data
    val_preds = np.zeros((len(y_val), n_models))
    for i, model in enumerate(models):
        val_preds[:, i] = model.predict(X_val)
    
    # For test data
    test_preds = np.zeros((len(y_test), n_models))
    individual_results = []
    individual_prediction_times = []
    
    # Measure prediction time for individual models
    for i, model in enumerate(models):
        # Measure prediction time on 1000 samples or whole test set if smaller
        sample_size = min(1000, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        # Repeat predictions for more accurate timing
        n_repeat = 10
        times = []
        for _ in range(n_repeat):
            start_time = time.time()
            model.predict(X_sample)
            pred_time = (time.time() - start_time) / sample_size  # Per sample time
            times.append(pred_time)
        
        avg_time = np.mean(times)
        individual_prediction_times.append(avg_time)
        
        # Make actual predictions for the full test set
        pred = model.predict(X_test)
        test_preds[:, i] = pred
        results = evaluate_prediction(y_test, pred, f"Model {i+1} on Test")
        results['prediction_time'] = avg_time * 1000  # Store in ms
        individual_results.append(results)
    
    # Create uniform ensemble predictions
    uniform_weights = np.ones(n_models) / n_models
    
    train_ensemble_pred = np.mean(train_preds, axis=1)
    val_ensemble_pred = np.mean(val_preds, axis=1)
    test_ensemble_pred = np.mean(test_preds, axis=1)
    
    # Measure ensemble prediction time
    sample_size = min(1000, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    # Define ensemble predict function for timing
    def ensemble_predict(X_data):
        ensemble_preds = np.zeros(len(X_data))
        for model in models:
            ensemble_preds += model.predict(X_data)
        return ensemble_preds / len(models)
    
    # Repeat predictions for more accurate timing
    n_repeat = 10
    ensemble_pred_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        ensemble_predict(X_sample)
        pred_time = (time.time() - start_time) / sample_size  # Per sample time
        ensemble_pred_times.append(pred_time)
    
    ensemble_pred_time = np.mean(ensemble_pred_times) * 1000  # Convert to ms
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Evaluate ensemble
    print("\nUniform Ensemble Evaluation:")
    train_results = evaluate_prediction(y_train, train_ensemble_pred, "Training")
    val_results = evaluate_prediction(y_val, val_ensemble_pred, "Validation")
    test_results = evaluate_prediction(y_test, test_ensemble_pred, "Test")
    
    # Add timing information
    test_results['prediction_time'] = ensemble_pred_time
    test_results['training_time'] = total_training_time
    
    # Save the ensemble model
    ensemble_model = {
        'weights': uniform_weights,
        'n_models': n_models
    }
    
    joblib.dump(ensemble_model, 'models/ensemble/uniform_ensemble.pkl')
    print("\nUniform ensemble model saved to 'models/ensemble/uniform_ensemble.pkl'")
    
    # Compare with individual models
    print("\nUniform Ensemble vs Best Individual Model:")
    metrics = ['rmse', 'r2', 'mae', 'mape']
    comparison = {}
    
    for metric in metrics:
        # Get individual metrics
        individual_metrics = [result[metric] for result in individual_results]
        
        # Find best individual model for this metric
        if metric == 'r2':  # For R², higher is better
            best_idx = np.argmax(individual_metrics)
            best_val = np.max(individual_metrics)
            improvement = ((test_results[metric] / best_val) - 1) * 100
            is_better = test_results[metric] > best_val
        else:  # For error metrics, lower is better
            best_idx = np.argmin(individual_metrics)
            best_val = np.min(individual_metrics)
            improvement = ((best_val / test_results[metric]) - 1) * 100
            is_better = test_results[metric] < best_val
        
        comparison[metric] = {
            'best_individual': best_val,
            'best_individual_idx': best_idx,
            'ensemble': test_results[metric],
            'improvement': improvement,
            'is_ensemble_better': is_better
        }
        
        # Print comparison
        sign = "+" if improvement > 0 else ""
        print(f"{metric.upper()}: Ensemble = {test_results[metric]:.4f}, " +
              f"Best Individual = {best_val:.4f} (Model {best_idx+1}), " +
              f"Improvement: {sign}{improvement:.2f}%")
    
    # Print timing information
    print("\nTiming Information:")
    print(f"Training Time: {total_training_time:.4f} s")
    print(f"Prediction Time: {ensemble_pred_time:.4f} ms/sample")
    print(f"Individual model prediction times: {[f'{t*1000:.4f} ms' for t in individual_prediction_times]}")
    
    # Create comparison plots
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        data = comparison[metric]
        
        # Individual model metrics
        individual_metrics = [result[metric] for result in individual_results]
        
        # Add ensemble as the last bar
        all_metrics = individual_metrics + [data['ensemble']]
        labels = [f"Model {j+1}" for j in range(n_models)] + ["Uniform\nEnsemble"]
        
        # Highlight the best performer
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
    plt.savefig('results/ensemble/uniform_ensemble_comparison.png')
    plt.close()
    
    # Create performance vs. speed plot
    plt.figure(figsize=(10, 6))
    
    # Get RMSE and prediction times
    rmse_values = [result['rmse'] for result in individual_results] + [test_results['rmse']]
    pred_times = [result['prediction_time'] for result in individual_results] + [ensemble_pred_time]
    
    # Create scatter plot
    plt.scatter(pred_times, rmse_values, alpha=0.7, s=100)
    
    # Add labels for each point
    for i, (time_val, rmse) in enumerate(zip(pred_times, rmse_values)):
        label = "Ensemble" if i == len(pred_times) - 1 else f"Model {i+1}"
        plt.annotate(label, (time_val, rmse), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('Prediction Time (ms/sample)')
    plt.ylabel('RMSE ($)')
    plt.title('Performance vs. Speed Trade-off')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/ensemble/performance_vs_speed.png')
    plt.close()
    
    # Plot predictions scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_ensemble_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_test.max(), test_ensemble_pred.max())
    min_val = min(y_test.min(), test_ensemble_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Uniform Ensemble Predicted Price')
    plt.title('Uniform Ensemble: Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('results/ensemble/uniform_ensemble_predictions.png')
    plt.close()
    
    # Save detailed predictions
    ensemble_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_ensemble_pred,
        'Error': y_test - test_ensemble_pred,
        'Absolute_Error': np.abs(y_test - test_ensemble_pred),
        'Percentage_Error': np.abs((y_test - test_ensemble_pred) / y_test) * 100
    })
    ensemble_df.to_csv('results/ensemble/uniform_ensemble_predictions.csv', index=False)
    
    # Save error distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ensemble_df['Error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Uniform Ensemble Prediction Error Distribution')
    plt.tight_layout()
    plt.savefig('results/ensemble/uniform_ensemble_error_distribution.png')
    plt.close()
    
    # Save detailed results
    with open('results/ensemble/uniform_ensemble_results.txt', 'w') as f:
        f.write("Uniform Ensemble Model Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"RMSE: ${test_results['rmse']:.2f}\n")
        f.write(f"R² Score: {test_results['r2']:.4f}\n")
        f.write(f"MAE: ${test_results['mae']:.2f}\n")
        f.write(f"MAPE: {test_results['mape']:.2f}%\n")
        f.write(f"Median AE: ${test_results['medae']:.2f}\n")
        f.write(f"Maximum Error: ${test_results['max_error']:.2f}\n")
        f.write(f"95% Prediction Interval: ${test_results['interval_95']:.2f}\n\n")
        
        f.write("Timing Metrics:\n")
        f.write(f"Training Time: {test_results['training_time']:.4f} s\n")
        f.write(f"Prediction Time: {test_results['prediction_time']:.4f} ms/sample\n\n")
        
        f.write("Comparison with Individual Models:\n")
        f.write("-"*50 + "\n")
        for metric, data in comparison.items():
            sign = "+" if data['improvement'] > 0 else ""
            f.write(f"{metric.upper()}: Ensemble = {data['ensemble']:.4f}, " +
                    f"Best Individual = {data['best_individual']:.4f} (Model {data['best_individual_idx']+1}), " +
                    f"Improvement: {sign}{data['improvement']:.2f}%\n")
        
        f.write("\nIndividual Model Performance:\n")
        f.write("-"*50 + "\n")
        for i, result in enumerate(individual_results):
            f.write(f"Model {i+1}:\n")
            f.write(f"RMSE: ${result['rmse']:.2f}\n")
            f.write(f"R² Score: {result['r2']:.4f}\n")
            f.write(f"MAE: ${result['mae']:.2f}\n")
            f.write(f"MAPE: {result['mape']:.2f}%\n")
            f.write(f"Prediction Time: {result['prediction_time']:.4f} ms/sample\n\n")
    
    # Calculate additional metrics
    print("\nCalculating additional metrics for model comparison...")
    ensemble_add_metrics, individual_add_metrics = calculate_additional_metrics(
        models, X_test, y_test, feature_names
    )
    
    # Update the text report with additional metrics
    with open('results/ensemble/uniform_ensemble_results.txt', 'a') as f:
        f.write("\nADDITIONAL MODEL EVALUATION METRICS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Uniform Ensemble Additional Metrics:\n")
        f.write(f"Adjusted R²: {ensemble_add_metrics['adj_r2']:.4f}\n")
        f.write(f"CV R² Mean: {ensemble_add_metrics['cv_r2_mean']:.4f} (±{ensemble_add_metrics['cv_r2_std']:.4f})\n")
        f.write(f"CV RMSE Mean: ${ensemble_add_metrics['cv_rmse_mean']:.2f} (±${ensemble_add_metrics['cv_rmse_std']:.2f})\n")
        f.write(f"Prediction Time (CV): {ensemble_add_metrics['prediction_time_ms']:.4f} ms/sample\n\n")
        
        f.write("Individual Models Additional Metrics:\n")
        f.write("-"*50 + "\n")
        for metrics in individual_add_metrics:
            i = metrics['model_idx']
            f.write(f"Model {i+1}:\n")
            f.write(f"Adjusted R²: {metrics['adj_r2']:.4f}\n")
            f.write(f"CV R² Mean: {metrics['cv_r2_mean']:.4f} (±{metrics['cv_r2_std']:.4f})\n")
            f.write(f"CV RMSE Mean: ${metrics['cv_rmse_mean']:.2f} (±${metrics['cv_rmse_std']:.2f})\n")
            f.write(f"Prediction Time (CV): {metrics['prediction_time_ms']:.4f} ms/sample\n\n")
    
    # Create visualization for additional metrics
    plt.figure(figsize=(15, 10))
    
    # Plot for Adjusted R²
    plt.subplot(2, 2, 1)
    model_labels = [f"Model {i+1}" for i in range(len(models))] + ["Uniform\nEnsemble"]
    adj_r2_values = [m['adj_r2'] for m in individual_add_metrics] + [ensemble_add_metrics['adj_r2']]
    
    # Determine best model
    best_idx = np.argmax(adj_r2_values)
    colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(adj_r2_values))]
    
    plt.bar(model_labels, adj_r2_values, color=colors)
    plt.title('Adjusted R² Comparison')
    plt.ylabel('Adjusted R²')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(adj_r2_values):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
    
    # Plot for CV R² Mean
    plt.subplot(2, 2, 2)
    cv_r2_values = [m['cv_r2_mean'] for m in individual_add_metrics] + [ensemble_add_metrics['cv_r2_mean']]
    cv_r2_std = [m['cv_r2_std'] for m in individual_add_metrics] + [ensemble_add_metrics['cv_r2_std']]
    
    # Determine best model
    best_idx = np.argmax(cv_r2_values)
    colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(cv_r2_values))]
    
    plt.bar(model_labels, cv_r2_values, color=colors, yerr=cv_r2_std, capsize=5)
    plt.title('Cross-Validated R² Comparison')
    plt.ylabel('CV R² Mean')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(cv_r2_values):
        plt.text(i, v + 0.005 + cv_r2_std[i], f"{v:.4f}", ha='center')
    
    # Plot for CV RMSE Mean
    plt.subplot(2, 2, 3)
    cv_rmse_values = [m['cv_rmse_mean'] for m in individual_add_metrics] + [ensemble_add_metrics['cv_rmse_mean']]
    cv_rmse_std = [m['cv_rmse_std'] for m in individual_add_metrics] + [ensemble_add_metrics['cv_rmse_std']]
    
    # Determine best model (lowest RMSE)
    best_idx = np.argmin(cv_rmse_values)
    colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(cv_rmse_values))]
    
    plt.bar(model_labels, cv_rmse_values, color=colors, yerr=cv_rmse_std, capsize=5)
    plt.title('Cross-Validated RMSE Comparison')
    plt.ylabel('CV RMSE Mean ($)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(cv_rmse_values):
        plt.text(i, v + cv_rmse_std[i] + 1000, f"${v:.2f}", ha='center')
    
    # Plot showing prediction times
    plt.subplot(2, 2, 4)
    pred_time_values = [m['prediction_time_ms'] for m in individual_add_metrics] + [ensemble_add_metrics['prediction_time_ms']]
    
    # Lower is better
    best_idx = np.argmin(pred_time_values)
    colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(pred_time_values))]
    
    plt.bar(model_labels, pred_time_values, color=colors)
    plt.title('Prediction Time (Lower is Better)')
    plt.ylabel('Prediction Time (ms/sample)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(pred_time_values):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('results/ensemble/additional_metrics_comparison.png')
    plt.close()
    
    # Add metrics to the results dictionary
    ensemble_results = {
        'train_results': train_results,
        'val_results': val_results,
        'test_results': test_results,
        'individual_results': individual_results,
        'comparison': comparison,
        'weights': uniform_weights,
        'timing': {
            'training_time': total_training_time,
            'prediction_time': ensemble_pred_time,
            'individual_prediction_times': individual_prediction_times
        },
        'additional_metrics': {
            'ensemble': ensemble_add_metrics,
            'individual': individual_add_metrics
        }
    }
    
    return ensemble_results

# Move this function outside of main() to the module level
def predict_with_ensemble(X, models_path='models/kfold', ensemble_path='models/ensemble/uniform_ensemble.pkl'):
    """Function to make predictions with the uniform ensemble"""
    # Load models
    models = []
    n_folds = 5
    for i in range(1, n_folds + 1):
        model_path = f'{models_path}/lightgbm_fold{i}.pkl'
        if os.path.exists(model_path):
            models.append(joblib.load(model_path))
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Get individual predictions
    predictions = np.zeros((X.shape[0], len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X)
    
    # Average the predictions
    ensemble_pred = np.mean(predictions, axis=1)
    return ensemble_pred

def main():
    print("\n" + "="*50)
    print("UNIFORM ENSEMBLE BUILDING")
    print("="*50)
    
    # Check if K-fold models exist
    kfold_paths = ['models/kfold/lightgbm_fold1.pkl', 'models/kfold/lightgbm_fold2.pkl']
    if not all(os.path.exists(path) for path in kfold_paths):
        print("K-fold models not found. Please run LightGBM_kfold.py first.")
        return
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data_with_top_features(
        "../../Dataset/processed_Resaleflatprices_XGB.csv", 
        top_features_count=100
    )
    
    # Load models
    n_folds = 5
    models = load_kfold_models(n_folds)
    
    if models is None or len(models) != n_folds:
        print(f"Error: Could not load all {n_folds} K-fold models")
        return
    
    # Build and evaluate uniform ensemble
    ensemble_results = build_uniform_ensemble(
        models, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Save the prediction function for later use
    joblib.dump(predict_with_ensemble, 'models/ensemble/predict_with_uniform_ensemble.pkl')
    
    print("\nUniform Ensemble Building completed!")
    print("Results saved in 'results/ensemble/' directory")
    print("Prediction function saved as 'models/ensemble/predict_with_uniform_ensemble.pkl'")
    print("\nAdditional metrics (Adjusted R², CV R², CV RMSE) have been computed and saved.")
    print(f"Training time: {ensemble_results['timing']['training_time']:.4f} seconds")
    print(f"Prediction time: {ensemble_results['timing']['prediction_time']:.4f} ms/sample")

if __name__ == "__main__":
    main()
