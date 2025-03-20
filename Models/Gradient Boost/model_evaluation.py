import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')
try:
    from meta_stacking import StackingPredictor
except ImportError:
    print("Warning: Could not import StackingPredictor from meta_stacking.py")
    print("Stacking model evaluation will be skipped.")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Create directories for results
os.makedirs('results/evaluation', exist_ok=True)

def load_test_data(data_path, top_features_path=None, top_n_features=100):
    """Load and prepare test data consistently across models"""
    print(f"Loading test data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Target column
    target_col = 'resale_price'
    
    # Get the same test set used across all models
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Load top features if specified
    if top_features_path and os.path.exists(top_features_path):
        print(f"Using top {top_n_features} features from {top_features_path}")
        features_df = pd.read_csv(top_features_path)
        features = features_df.sort_values('Importance', ascending=False).head(top_n_features)['Feature'].tolist()
        # Remove target if it's in the feature list
        if target_col in features:
            features.remove(target_col)
    else:
        # Use all columns except target
        features = [col for col in df.columns if col != target_col]
    
    X_test = test[features]
    y_test = test[target_col]
    
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test, features

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return model performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'predictions': y_pred
    }
    
    print(f"{model_name} Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 40)
    
    return metrics

def load_single_models():
    """Load and evaluate individual models"""
    model_paths = {
        'XGBoost (Base)': 'models/xgboost_model.pkl',
        'LightGBM (Base)': 'models/lightgbm_model.pkl',
        'CatBoost (Base)': 'models/catboost_model.pkl'
    }
    
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: Could not find model at {path}")
    
    return models

def load_kfold_models():
    """Load K-fold models"""
    kfold_models = []
    for i in range(1, 6):  # Assume 5 folds
        path = f'models/kfold/lightgbm_fold{i}.pkl'
        if os.path.exists(path):
            kfold_models.append(joblib.load(path))
            print(f"Loaded K-fold model {i} from {path}")
        else:
            print(f"Warning: Could not find model at {path}")
    
    return kfold_models

def predict_with_uniform_ensemble(X, kfold_models):
    """Make predictions with uniform ensemble"""
    if not kfold_models:
        raise ValueError("No K-fold models provided for ensemble prediction")
    
    predictions = np.zeros(len(X))
    for model in kfold_models:
        predictions += model.predict(X)
    
    return predictions / len(kfold_models)

def load_stacking_predictor():
    """Load stacking meta-model predictor"""
    stacking_path = 'models/stacking/stacking_predictor.pkl'
    
    if os.path.exists(stacking_path):
        predictor = joblib.load(stacking_path)
        print(f"Loaded stacking predictor from {stacking_path}")
        return predictor
    else:
        print(f"Warning: Could not find stacking predictor at {stacking_path}")
        return None

def perform_model_evaluation(X_test, y_test):
    """Evaluate and compare all models"""
    print("\n" + "="*50)
    print("EVALUATING ALL MODELS ON TEST DATA")
    print("="*50)
    
    results = []
    predictions = {}
    
    # 1. Evaluate single models
    single_models = load_single_models()
    for name, model in single_models.items():
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000 / len(X_test)  # ms per sample
        
        metrics = evaluate_model(y_test, y_pred, name)
        metrics['inference_time'] = inference_time
        results.append(metrics)
        predictions[name] = y_pred
    
    # 2. Evaluate K-fold models individually
    kfold_models = load_kfold_models()
    if kfold_models:
        for i, model in enumerate(kfold_models):
            name = f"LightGBM Fold {i+1}"
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = (time.time() - start_time) * 1000 / len(X_test)
            
            metrics = evaluate_model(y_test, y_pred, name)
            metrics['inference_time'] = inference_time
            results.append(metrics)
            # ADDED: Store predictions for k-fold models too
            predictions[name] = y_pred
    
    # 3. Evaluate uniform ensemble
    if kfold_models:
        start_time = time.time()
        y_pred_ensemble = predict_with_uniform_ensemble(X_test, kfold_models)
        inference_time = (time.time() - start_time) * 1000 / len(X_test)
        
        metrics = evaluate_model(y_test, y_pred_ensemble, "Uniform Ensemble")
        metrics['inference_time'] = inference_time
        results.append(metrics)
        predictions["Uniform Ensemble"] = y_pred_ensemble
    
    # 4. Skip stacking model evaluation for simplicity
    print("\nSkipping Meta-Model Stacking evaluation.")
    
    return results, predictions

def create_comparison_visualizations(results, predictions, y_test):
    """Create visualizations comparing model performance"""
    
    # 1. Create metrics comparison chart
    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'predictions'} 
        for r in results
    ])
    
    # Sort by RMSE (ascending)
    metrics_df = metrics_df.sort_values('rmse')
    
    # 2. Plot metrics comparison
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE
    sns.barplot(x='model', y='rmse', data=metrics_df, ax=axs[0, 0], palette='viridis')
    axs[0, 0].set_title('RMSE (Lower is Better)')
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
    axs[0, 0].set_ylabel('RMSE ($)')
    
    # R²
    sns.barplot(x='model', y='r2', data=metrics_df, ax=axs[0, 1], palette='viridis')
    axs[0, 1].set_title('R² Score (Higher is Better)')
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
    axs[0, 1].set_ylabel('R² Score')
    
    # MAE
    sns.barplot(x='model', y='mae', data=metrics_df, ax=axs[1, 0], palette='viridis')
    axs[1, 0].set_title('Mean Absolute Error (Lower is Better)')
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
    axs[1, 0].set_ylabel('MAE ($)')
    
    # MAPE
    sns.barplot(x='model', y='mape', data=metrics_df, ax=axs[1, 1], palette='viridis')
    axs[1, 1].set_title('Mean Absolute Percentage Error (Lower is Better)')
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
    axs[1, 1].set_ylabel('MAPE (%)')
    
    plt.tight_layout()
    plt.savefig('results/evaluation/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots of predictions for top models
    # Only use models that have predictions stored
    available_models = list(predictions.keys())
    # Find top performing models that have predictions available
    top_models = [model for model in metrics_df['model'] if model in available_models][:3]
    
    if top_models:
        fig, axs = plt.subplots(1, len(top_models), figsize=(18, 6))
        
        # Handle case with only one model
        if len(top_models) == 1:
            axs = [axs]
            
        for i, model_name in enumerate(top_models):
            y_pred = predictions[model_name]
            
            axs[i].scatter(y_test, y_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            axs[i].set_title(f'{model_name}')
            axs[i].set_xlabel('Actual Price ($)')
            if i == 0:
                axs[i].set_ylabel('Predicted Price ($)')
            
            # Add R² text
            r2_val = metrics_df[metrics_df['model'] == model_name]['r2'].values[0]
            rmse_val = metrics_df[metrics_df['model'] == model_name]['rmse'].values[0]
            axs[i].text(0.05, 0.95, f"R² = {r2_val:.4f}\nRMSE = ${rmse_val:.2f}",
                     transform=axs[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/evaluation/top_models_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Warning: No models with predictions available for scatter plots")
    
    # 4. Residual Analysis for best model
    # Use the best model that has predictions available
    best_model = None
    for model in metrics_df['model']:
        if model in predictions:
            best_model = model
            break
            
    if best_model:
        best_pred = predictions[best_model]
        
        residuals = y_test - best_pred
        percentage_error = 100 * np.abs(residuals) / y_test
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # Residuals histogram
        axs[0].hist(residuals, bins=50, alpha=0.7)
        axs[0].axvline(x=0, color='r', linestyle='--')
        axs[0].set_title(f'Residual Distribution - {best_model}')
        axs[0].set_xlabel('Residual (Actual - Predicted)')
        axs[0].set_ylabel('Frequency')
        
        # Residuals vs Predicted
        axs[1].scatter(best_pred, residuals, alpha=0.6)
        axs[1].axhline(y=0, color='r', linestyle='--')
        axs[1].set_title(f'Residuals vs Predicted - {best_model}')
        axs[1].set_xlabel('Predicted Price ($)')
        axs[1].set_ylabel('Residual ($)')
        
        # Percentage error distribution
        axs[2].hist(percentage_error, bins=50, alpha=0.7)
        axs[2].set_title(f'Percentage Error Distribution - {best_model}')
        axs[2].set_xlabel('Absolute Percentage Error (%)')
        axs[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('results/evaluation/best_model_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Warning: No best model with predictions available for residual analysis")
    
    # 5. Model improvement progression
    progression_models = [
        "XGBoost (Base)",
        "LightGBM (Base)",
        "LightGBM Fold 1",
        "Uniform Ensemble"
    ]
    
    # Filter to models that exist in results
    progression_models = [m for m in progression_models if m in metrics_df['model'].values]
    
    if len(progression_models) >= 2:
        progress_df = metrics_df[metrics_df['model'].isin(progression_models)]
        baseline_rmse = progress_df[progress_df['model'] == progression_models[0]]['rmse'].values[0]
        
        progress_df['improvement'] = ((baseline_rmse - progress_df['rmse']) / baseline_rmse) * 100
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y='improvement', data=progress_df, palette='viridis')
        plt.title('Model Improvement Progression (% RMSE Reduction from Baseline)')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.2f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/evaluation/model_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Inference time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='inference_time', data=metrics_df, palette='viridis')
    plt.title('Model Inference Time Comparison')
    plt.ylabel('Time per Sample (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/evaluation/inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # IMPORTANT: Return the metrics DataFrame for use in save_comparison_summary
    return metrics_df

def save_comparison_summary(metrics_df, y_test, predictions):
    """Save detailed comparison results to file"""
    # Sort by RMSE (best first)
    metrics_df = metrics_df.sort_values('rmse')
    
    # Save metrics to CSV
    metrics_df.to_csv('results/evaluation/model_comparison_metrics.csv', index=False)
    
    # Create detailed report
    with open('results/evaluation/model_comparison_report.txt', 'w') as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("PERFORMANCE RANKING (BY RMSE)\n")
        f.write("-"*50 + "\n")
        
        for i, row in metrics_df.iterrows():
            f.write(f"{i+1}. {row['model']}\n")
            f.write(f"   RMSE: ${row['rmse']:.2f}\n")
            f.write(f"   R² Score: {row['r2']:.4f}\n")
            f.write(f"   MAE: ${row['mae']:.2f}\n")
            f.write(f"   MAPE: {row['mape']:.2f}%\n")
            f.write(f"   Inference Time: {row['inference_time']:.4f} ms/sample\n\n")
        
        # Calculate improvements from baseline
        if len(metrics_df) > 1:
            baseline_model = metrics_df.iloc[-1]['model']
            baseline_rmse = metrics_df.iloc[-1]['rmse']
            best_model = metrics_df.iloc[0]['model']
            best_rmse = metrics_df.iloc[0]['rmse']
            
            improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
            
            f.write("\nIMPROVEMENT ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write(f"Baseline Model: {baseline_model}\n")
            f.write(f"Best Model: {best_model}\n")
            f.write(f"RMSE Improvement: {improvement:.2f}%\n\n")
            
        # Analysis of best model errors
        # FIXED: Find the best model that has predictions available
        best_model = None
        for model in metrics_df['model']:
            if model in predictions:
                best_model = model
                break
                
        if best_model:
            best_pred = predictions[best_model]
            
            residuals = y_test - best_pred
            abs_errors = np.abs(residuals)
            pct_errors = 100 * abs_errors / y_test
            
            f.write("\nBEST MODEL ERROR ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write(f"Model: {best_model}\n\n")
            
            f.write("Error Percentiles:\n")
            for p in [50, 75, 90, 95, 99]:
                f.write(f"  {p}th Percentile Absolute Error: ${np.percentile(abs_errors, p):.2f}\n")
                f.write(f"  {p}th Percentile Percentage Error: {np.percentile(pct_errors, p):.2f}%\n")
            
            f.write("\nPrice Band Analysis:\n")
            price_bands = [
                (0, 300000, "< $300K"),
                (300000, 500000, "$300K-$500K"),
                (500000, 700000, "$500K-$700K"),
                (700000, 1000000, "$700K-$1M"),
                (1000000, float('inf'), "> $1M")
            ]
            
            for low, high, label in price_bands:
                mask = (y_test >= low) & (y_test < high)
                if sum(mask) > 0:
                    band_rmse = np.sqrt(mean_squared_error(y_test[mask], best_pred[mask])) if sum(mask) > 0 else 0
                    band_mape = np.mean(np.abs((y_test[mask] - best_pred[mask]) / y_test[mask])) * 100 if sum(mask) > 0 else 0
                    band_count = sum(mask)
                    
                    f.write(f"\n  {label} ({band_count} properties):\n")
                    f.write(f"    RMSE: ${band_rmse:.2f}\n")
                    f.write(f"    MAPE: {band_mape:.2f}%\n")
            
            f.write("\n\nCONCLUSION AND RECOMMENDATIONS\n")
            f.write("-"*50 + "\n")
            
            # Automatically generate recommendations based on results
            if best_model == "Uniform Ensemble":
                f.write("The Uniform Ensemble model performs best and is recommended for deployment.\n")
                f.write("Benefits include superior accuracy, good generalization, and implementation simplicity.\n")
            elif "Stacking" in best_model:
                f.write("The Meta-Model Stacking approach performs best and is recommended for deployment.\n")
                f.write("While more complex to implement, it provides the highest accuracy according to test metrics.\n")
            else:
                f.write(f"The {best_model} performs best and is recommended for deployment.\n")
            
            # Add inference time considerations
            fastest_model = metrics_df.iloc[metrics_df['inference_time'].idxmin()]['model']
            if fastest_model != best_model:
                f.write(f"\nNOTE: If inference speed is critical, consider {fastest_model}, which is the fastest model.\n")
                f.write(f"There would be a trade-off of {((metrics_df[metrics_df['model']==fastest_model]['rmse'].values[0] - metrics_df[metrics_df['model']==best_model]['rmse'].values[0])/metrics_df[metrics_df['model']==best_model]['rmse'].values[0])*100:.2f}% in RMSE performance.\n")
        else:
            f.write("\nWARNING: No best model with predictions available for detailed analysis.\n")

    print(f"Detailed comparison report saved to 'results/evaluation/model_comparison_report.txt'")
    
    # Return best model name (or None if no predictions available)
    return best_model if best_model else metrics_df.iloc[0]['model']

def main():
    print("\n" + "="*50)
    print("STEP 8: COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    # Load test data
    X_test, y_test, features = load_test_data(
        data_path="../../Dataset/processed_Resaleflatprices_XGB.csv",
        top_features_path="feature_analysis/xgboost_importance.csv",
        top_n_features=100
    )
    
    # Evaluate all models
    results, predictions = perform_model_evaluation(X_test, y_test)
    
    # Create visualizations
    metrics_df = create_comparison_visualizations(results, predictions, y_test)
    
    # Save detailed comparison summary
    best_model = save_comparison_summary(metrics_df, y_test, predictions)
    
    print("\nModel evaluation completed!")
    print(f"Best performing model: {best_model}")
    print("Visualizations and detailed report saved to 'results/evaluation/' directory")

if __name__ == "__main__":
    main()
