import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create directories for models and results
os.makedirs('models/stacking', exist_ok=True)
os.makedirs('results/stacking', exist_ok=True)

def load_data_with_top_features(data_path, top_features_count=100):
    """
    Load data and filter to only use top features - consistent with previous scripts
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

def evaluate_model(y_true, y_pred, dataset_name=""):
    """Evaluate model performance"""
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Create a results dictionary
    results = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'predictions': y_pred
    }
    
    # Print metrics if dataset_name is provided
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
    
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

def generate_base_predictions(models, X_train, X_val, X_test):
    """Generate predictions from base models for meta-model input"""
    print("Generating base model predictions for meta-model features...")
    n_models = len(models)
    
    # For training data
    train_preds = np.zeros((len(X_train), n_models))
    for i, model in enumerate(models):
        print(f"Generating training predictions for Model {i+1}")
        train_preds[:, i] = model.predict(X_train)
    
    # For validation data
    val_preds = np.zeros((len(X_val), n_models))
    for i, model in enumerate(models):
        print(f"Generating validation predictions for Model {i+1}")
        val_preds[:, i] = model.predict(X_val)
    
    # For test data
    test_preds = np.zeros((len(X_test), n_models))
    for i, model in enumerate(models):
        print(f"Generating test predictions for Model {i+1}")
        test_preds[:, i] = model.predict(X_test)
    
    return train_preds, val_preds, test_preds

def create_meta_features(X, model_preds, include_original_features=True, top_features=None):
    """
    Create meta-features by combining original features with model predictions
    
    Parameters:
    -----------
    X : DataFrame
        Original features
    model_preds : array
        Predictions from base models
    include_original_features : bool
        Whether to include original features alongside model predictions
    top_features : list
        If specified, only include these original features
    
    Returns:
    --------
    meta_features : array
        Combined features for meta-model
    """
    # Convert model predictions to DataFrame
    pred_df = pd.DataFrame(
        model_preds, 
        columns=[f'model_{i+1}_pred' for i in range(model_preds.shape[1])]
    )
    
    # Add statistics about the predictions
    pred_df['pred_mean'] = pred_df.mean(axis=1)
    pred_df['pred_median'] = pred_df.median(axis=1)
    pred_df['pred_std'] = pred_df.std(axis=1)
    pred_df['pred_min'] = pred_df.min(axis=1)
    pred_df['pred_max'] = pred_df.max(axis=1)
    pred_df['pred_range'] = pred_df['pred_max'] - pred_df['pred_min']
    
    # Calculate residuals between models
    if model_preds.shape[1] > 1:
        for i in range(model_preds.shape[1]):
            for j in range(i+1, model_preds.shape[1]):
                pred_df[f'residual_{i+1}_{j+1}'] = model_preds[:, i] - model_preds[:, j]
    
    if include_original_features:
        if top_features is not None and len(top_features) > 0:
            # Select only specified top features
            selected_features = X[top_features].reset_index(drop=True)
        else:
            # Use all original features
            selected_features = X.reset_index(drop=True)
        
        # Combine with model predictions
        meta_features = pd.concat([selected_features, pred_df.reset_index(drop=True)], axis=1)
    else:
        meta_features = pred_df
        
    return meta_features

def train_meta_models(meta_X_train, y_train, meta_X_val, y_val):
    """Train various meta-models and select the best one"""
    print("\n" + "="*50)
    print("Training and Selecting Meta-Models")
    print("="*50)
    
    meta_models = {
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.001, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    # Train and evaluate each meta-model
    model_results = {}
    
    for name, model in meta_models.items():
        print(f"\nTraining {name} meta-model...")
        start_time = time.time()
        
        # Train the model
        model.fit(meta_X_train, y_train)
        
        # Evaluate on validation set
        val_pred = model.predict(meta_X_val)
        val_results = evaluate_model(y_val, val_pred, f"{name} on Validation")
        
        # Store results
        training_time = time.time() - start_time
        model_results[name] = {
            'model': model,
            'val_results': val_results,
            'training_time': training_time
        }
    
    # Find the best meta-model based on validation RMSE
    best_rmse = float('inf')
    best_model_name = None
    
    for name, results in model_results.items():
        if results['val_results']['rmse'] < best_rmse:
            best_rmse = results['val_results']['rmse']
            best_model_name = name
    
    # Extract the best model
    best_model = model_results[best_model_name]['model']
    print(f"\nBest Meta-Model: {best_model_name}")
    print(f"Validation RMSE: ${best_rmse:.2f}")
    
    # Save all trained models
    for name, results in model_results.items():
        joblib.dump(results['model'], f'models/stacking/meta_{name.lower()}.pkl')
        
    # Save model results for comparison
    val_metrics = {name: results['val_results'] for name, results in model_results.items()}
    
    return best_model, best_model_name, model_results

def evaluate_stacking_model(best_meta_model, base_models, meta_X_train, y_train, 
                         meta_X_val, y_val, meta_X_test, y_test, X_test,  # Added X_test parameter
                         uniform_ensemble_preds=None):
    """Evaluate the stacking model and compare with uniform ensemble"""
    print("\n" + "="*50)
    print("Evaluating Stacking Model")
    print("="*50)
    
    # Make predictions with the best meta-model
    train_pred = best_meta_model.predict(meta_X_train)
    val_pred = best_meta_model.predict(meta_X_val)
    test_pred = best_meta_model.predict(meta_X_test)
    
    # Evaluate the meta-model
    train_results = evaluate_model(y_train, train_pred, "Meta-Model on Training")
    val_results = evaluate_model(y_val, val_pred, "Meta-Model on Validation")
    test_results = evaluate_model(y_test, test_pred, "Meta-Model on Test")
    
    # Get base model individual performances on test set
    base_model_results = []
    for i, model in enumerate(base_models):
        base_pred = model.predict(X_test)
        results = evaluate_model(y_test, base_pred, f"Base Model {i+1} on Test")
        base_model_results.append(results)
    
    # Calculate uniform ensemble predictions if not provided
    if uniform_ensemble_preds is None:
        uniform_preds = np.zeros_like(test_pred)
        for model in base_models:
            uniform_preds += model.predict(X_test)
        uniform_preds /= len(base_models)
    else:
        uniform_preds = uniform_ensemble_preds
    
    # Evaluate uniform ensemble
    uniform_results = evaluate_model(y_test, uniform_preds, "Uniform Ensemble on Test")
    
    # Compare meta-model with uniform ensemble and best base model
    print("\nComparison - Meta-Model vs. Uniform Ensemble vs. Best Base Model:")
    
    # Find best base model
    best_base_rmse = float('inf')
    best_base_idx = 0
    for i, result in enumerate(base_model_results):
        if result['rmse'] < best_base_rmse:
            best_base_rmse = result['rmse']
            best_base_idx = i
    
    best_base_results = base_model_results[best_base_idx]
    
    # Compare metrics
    metrics = ['rmse', 'r2', 'mae', 'mape']
    comparison = {}
    
    for metric in metrics:
        # Calculate improvements
        if metric == 'r2':  # For R², higher is better
            meta_vs_uniform = ((test_results[metric] / uniform_results[metric]) - 1) * 100
            meta_vs_base = ((test_results[metric] / best_base_results[metric]) - 1) * 100
        else:  # For error metrics, lower is better
            meta_vs_uniform = ((uniform_results[metric] / test_results[metric]) - 1) * 100
            meta_vs_base = ((best_base_results[metric] / test_results[metric]) - 1) * 100
            
        comparison[metric] = {
            'meta': test_results[metric],
            'uniform': uniform_results[metric],
            'best_base': best_base_results[metric],
            'best_base_idx': best_base_idx,
            'meta_vs_uniform': meta_vs_uniform,
            'meta_vs_base': meta_vs_base
        }
        
        # Print comparison
        print(f"{metric.upper()}:")
        print(f"  Meta-Model: {test_results[metric]:.4f}")
        print(f"  Uniform Ensemble: {uniform_results[metric]:.4f}")
        print(f"  Best Base Model: {best_base_results[metric]:.4f} (Model {best_base_idx+1})")
        
        sign_uniform = "+" if meta_vs_uniform > 0 else ""
        sign_base = "+" if meta_vs_base > 0 else ""
        print(f"  Improvement over Uniform: {sign_uniform}{meta_vs_uniform:.2f}%")
        print(f"  Improvement over Best Base: {sign_base}{meta_vs_base:.2f}%")
        print()
    
    # Create comparison chart
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        data = comparison[metric]
        
        labels = ['Meta-Model', 'Uniform\nEnsemble', f'Best Base\nModel {best_base_idx+1}']
        values = [data['meta'], data['uniform'], data['best_base']]
        
        # Determine best value
        if metric == 'r2':  # Higher is better for R²
            best_idx = np.argmax(values)
        else:  # Lower is better for error metrics
            best_idx = np.argmin(values)
            
        colors = ['#ff9999' if i != best_idx else '#66b3ff' for i in range(len(values))]
        
        plt.bar(labels, values, color=colors)
        
        # Add value labels
        for j, v in enumerate(values):
            if metric == 'mape':
                plt.text(j, v + 0.1, f"{v:.2f}%", ha='center')
            elif metric == 'r2':
                plt.text(j, v + 0.005, f"{v:.4f}", ha='center')
            else:
                plt.text(j, v + (max(values) * 0.01), f"${v:.2f}", ha='center')
                
        plt.title(f"{metric.upper()} Comparison")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/stacking/meta_model_comparison.png')
    plt.close()
    
    # Plot actual vs. predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_test.max(), test_pred.max())
    min_val = min(y_test.min(), test_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Meta-Model Predicted Price')
    plt.title('Meta-Model: Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('results/stacking/meta_model_predictions.png')
    plt.close()
    
    # Save test predictions
    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Meta_Model': test_pred,
        'Uniform_Ensemble': uniform_preds,
        'Error': y_test - test_pred,
        'Absolute_Error': np.abs(y_test - test_pred),
        'Percentage_Error': np.abs((y_test - test_pred) / y_test) * 100
    })
    
    # Add base model predictions
    for i, model in enumerate(base_models):
        prediction_df[f'Base_Model_{i+1}'] = model.predict(X_test)
    
    prediction_df.to_csv('results/stacking/meta_model_predictions.csv', index=False)
    
    # Save error distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_df['Error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Meta-Model Prediction Error Distribution')
    plt.tight_layout()
    plt.savefig('results/stacking/meta_model_error_distribution.png')
    plt.close()
    
    # Save detailed results
    with open('results/stacking/meta_model_results.txt', 'w') as f:
        f.write("Meta-Model Stacking Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"RMSE: ${test_results['rmse']:.2f}\n")
        f.write(f"R² Score: {test_results['r2']:.4f}\n")
        f.write(f"MAE: ${test_results['mae']:.2f}\n")
        f.write(f"MAPE: {test_results['mape']:.2f}%\n\n")
        
        f.write("Comparison with Uniform Ensemble and Best Base Model:\n")
        f.write("-"*50 + "\n")
        
        for metric, data in comparison.items():
            sign_uniform = "+" if data['meta_vs_uniform'] > 0 else ""
            sign_base = "+" if data['meta_vs_base'] > 0 else ""
            
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Meta-Model: {data['meta']:.4f}\n")
            f.write(f"  Uniform Ensemble: {data['uniform']:.4f}\n")
            f.write(f"  Best Base Model: {data['best_base']:.4f} (Model {data['best_base_idx']+1})\n")
            f.write(f"  Improvement over Uniform: {sign_uniform}{data['meta_vs_uniform']:.2f}%\n")
            f.write(f"  Improvement over Best Base: {sign_base}{data['meta_vs_base']:.2f}%\n\n")
    
    # Return all results
    return {
        'train_results': train_results,
        'val_results': val_results,
        'test_results': test_results,
        'uniform_results': uniform_results,
        'base_results': base_model_results,
        'comparison': comparison
    }

class StackingPredictor:
    """Class to make predictions with the stacking model"""
    
    def __init__(self, meta_model_path='models/stacking/meta_model.pkl', 
                 base_models_path='models/kfold', n_folds=5,
                 include_original_features=True, top_features_path=None):
        self.meta_model_path = meta_model_path
        self.base_models_path = base_models_path
        self.n_folds = n_folds
        self.meta_model = None
        self.base_models = []
        self.include_original_features = include_original_features
        self.top_features = None
        
        # Load top features if specified
        if top_features_path:
            features_df = pd.read_csv(top_features_path)
            self.top_features = features_df.head(20)['Feature'].tolist()
    
    def load_models(self):
        """Load all models from disk"""
        # Load base models
        self.base_models = []
        for i in range(1, self.n_folds + 1):
            model_path = f'{self.base_models_path}/lightgbm_fold{i}.pkl'
            if os.path.exists(model_path):
                self.base_models.append(joblib.load(model_path))
            else:
                raise FileNotFoundError(f"Base model not found at {model_path}")
        
        # Load meta-model
        if os.path.exists(self.meta_model_path):
            self.meta_model = joblib.load(self.meta_model_path)
        else:
            raise FileNotFoundError(f"Meta-model not found at {self.meta_model_path}")
            
        return self
    
    def predict(self, X):
        """Make predictions with the stacking model"""
        # Load models if not already loaded
        if not self.meta_model or not self.base_models:
            self.load_models()
            
        # Generate base model predictions
        base_preds = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict(X)
        
        # Create meta-features
        meta_features = create_meta_features(X, base_preds, 
                                           self.include_original_features,
                                           self.top_features)
        
        # Make final prediction with meta-model
        return self.meta_model.predict(meta_features)

def main():
    print("\n" + "="*50)
    print("META-MODEL STACKING (STEP 7)")
    print("="*50)
    
    # Check if base models exist
    base_model_path = 'models/kfold/lightgbm_fold1.pkl'
    if not os.path.exists(base_model_path):
        print("Base models not found. Please run LightGBM_kfold.py first.")
        return
    
    # Check if uniform ensemble results exist
    uniform_results_path = 'results/ensemble/uniform_ensemble_predictions.csv'
    uniform_preds = None
    if os.path.exists(uniform_results_path):
        print("Loading uniform ensemble predictions for comparison...")
        uniform_df = pd.read_csv(uniform_results_path)
        uniform_preds = uniform_df['Predicted'].values
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data_with_top_features(
        "../../Dataset/processed_Resaleflatprices_XGB.csv", 
        top_features_count=100
    )
    
    # Load base models
    print("\nLoading base models...")
    base_models = load_kfold_models(n_folds=5)
    
    if base_models is None or len(base_models) < 1:
        print("Error: Could not load base models")
        return
    
    # Generate predictions from base models
    print("\nGenerating features for meta-model...")
    train_preds, val_preds, test_preds = generate_base_predictions(
        base_models, X_train, X_val, X_test
    )
    
    # Decide which original features to keep for meta-model
    # Use top 20 features for better generalization
    features_path = 'feature_analysis/xgboost_importance.csv'
    top_features = None
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        top_features = features_df.sort_values('Importance', ascending=False).head(20)['Feature'].tolist()
    
    # Create meta-features by combining model predictions with original features
    meta_X_train = create_meta_features(X_train, train_preds, include_original_features=True, top_features=top_features)
    meta_X_val = create_meta_features(X_val, val_preds, include_original_features=True, top_features=top_features)
    meta_X_test = create_meta_features(X_test, test_preds, include_original_features=True, top_features=top_features)
    
    print(f"\nTraining meta-model on {meta_X_train.shape[1]} features")
    print(f"  {len(base_models)} base model predictions")
    print(f"  {len(top_features)} original features" if top_features else "  All original features")
    
    # Train and select meta-model
    best_meta_model, best_model_name, meta_model_results = train_meta_models(
        meta_X_train, y_train, meta_X_val, y_val
    )
    
    # Save the best model separately
    joblib.dump(best_meta_model, 'models/stacking/meta_model.pkl')
    print(f"Best meta-model ({best_model_name}) saved to 'models/stacking/meta_model.pkl'")
    
    # Evaluate the stacking model and compare to ensemble
    stacking_results = evaluate_stacking_model(
        best_meta_model, base_models, 
        meta_X_train, y_train, meta_X_val, y_val, meta_X_test, y_test,
        X_test,  # Add this parameter
        uniform_preds
    )

    
    # Save the predictor class
    predictor = StackingPredictor(
        meta_model_path='models/stacking/meta_model.pkl',
        include_original_features=True,
        top_features_path=features_path if os.path.exists(features_path) else None
    )
    
    joblib.dump(predictor, 'models/stacking/stacking_predictor.pkl')
    print("Stacking predictor saved to 'models/stacking/stacking_predictor.pkl'")
    
    # Usage instructions
    print("\nMeta-Model Stacking completed!")
    print("To use the stacking model for prediction:")
    print("  1. Load the predictor: predictor = joblib.load('models/stacking/stacking_predictor.pkl')")
    print("  2. Make predictions: predictions = predictor.predict(X_new)")

if __name__ == "__main__":
    main()
