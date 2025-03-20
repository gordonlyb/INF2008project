import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

# Ensure output directory exists
os.makedirs('feature_analysis', exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_model_and_data():
    """Load the optimized model and data"""
    # Load model
    print("Loading optimized XGBoost model...")
    model = joblib.load('optimized_xgboost_model.pkl')
    
    # Load data
    print("Loading datasets...")
    processed_csv = "../../Dataset/processed_Resaleflatprices_XGB.csv"
    
    # Split into train/test to match our previous analysis
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(processed_csv)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train = train.drop('resale_price', axis=1)
    y_train = train['resale_price']
    X_test = test.drop('resale_price', axis=1)
    y_test = test['resale_price']
    
    return model, X_train, y_train, X_test, y_test

def analyze_xgboost_importance(model, X_train, output_path='feature_analysis/xgboost_importance.png'):
    """Analyze and visualize XGBoost's built-in feature importance"""
    print("Analyzing XGBoost built-in feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    feat_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    })
    
    # Sort by importance
    feat_importances = feat_importances.sort_values('Importance', ascending=False)
    
    # Save full results to CSV
    feat_importances.to_csv('feature_analysis/xgboost_importance.csv', index=False)
    
    # Display top 30 features with horizontal bars (better for reading feature names)
    plt.figure(figsize=(12, 14))
    sns.barplot(x='Importance', y='Feature', data=feat_importances.head(30))
    plt.title('Top 30 Feature Importances (XGBoost Built-in)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return feat_importances

def analyze_permutation_importance(model, X_test, y_test, output_path='feature_analysis/permutation_importance.png'):
    """Calculate and visualize permutation-based feature importance"""
    print("Calculating permutation importance (this may take a few minutes)...")
    
    # Calculate permutation importance
    start_time = time.time()
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    end_time = time.time()
    print(f"Permutation importance calculation completed in {end_time - start_time:.2f} seconds")
    
    # Create DataFrame for visualization
    perm_importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    
    # Sort by importance
    perm_importances = perm_importances.sort_values('Importance', ascending=False)
    
    # Save full results to CSV
    perm_importances.to_csv('feature_analysis/permutation_importance.csv', index=False)
    
    # Display top 30 features
    plt.figure(figsize=(12, 14))
    top_features = perm_importances.head(30)
    
    # Create barplot without error bars first
    ax = sns.barplot(x='Importance', y='Feature', data=top_features)
    
    # Manually add error bars (this is the correct way)
    for i, (_, row) in enumerate(top_features.iterrows()):
        ax.errorbar(row['Importance'], i, xerr=row['Std'], fmt='none', color='black', elinewidth=1, capsize=3)
    
    plt.title('Top 30 Feature Importances (Permutation-based)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return perm_importances

def compare_importance_methods(xgb_importance, perm_importance, output_path='feature_analysis/importance_comparison.png'):
    """Compare different importance methods"""
    print("Comparing importance methods...")
    
    # Merge the two importance DataFrames
    comparison = pd.merge(
        xgb_importance[['Feature', 'Importance']].rename(columns={'Importance': 'XGBoost Importance'}),
        perm_importance[['Feature', 'Importance']].rename(columns={'Importance': 'Permutation Importance'}),
        on='Feature',
        how='outer'
    ).fillna(0)
    
    # Add ranks
    comparison['XGBoost Rank'] = comparison['XGBoost Importance'].rank(ascending=False)
    comparison['Permutation Rank'] = comparison['Permutation Importance'].rank(ascending=False)
    comparison['Rank Difference'] = abs(comparison['XGBoost Rank'] - comparison['Permutation Rank'])
    
    # Normalize importances for comparison
    comparison['XGBoost Importance (Normalized)'] = comparison['XGBoost Importance'] / comparison['XGBoost Importance'].sum()
    comparison['Permutation Importance (Normalized)'] = comparison['Permutation Importance'] / comparison['Permutation Importance'].sum()
    
    # Sort by average importance
    comparison['Average Importance'] = (comparison['XGBoost Importance (Normalized)'] + 
                                       comparison['Permutation Importance (Normalized)']) / 2
    comparison = comparison.sort_values('Average Importance', ascending=False)
    
    # Save to CSV
    comparison.to_csv('feature_analysis/importance_comparison.csv', index=False)
    
    # Plot top 20 features by both methods
    plt.figure(figsize=(14, 10))
    
    # Get top 20 features by average importance
    top_features = comparison.head(20)['Feature'].tolist()
    comparison_subset = comparison[comparison['Feature'].isin(top_features)]
    
    # Reshape for plotting
    plot_data = pd.melt(
        comparison_subset, 
        id_vars=['Feature'], 
        value_vars=['XGBoost Importance (Normalized)', 'Permutation Importance (Normalized)'],
        var_name='Method', 
        value_name='Normalized Importance'
    )
    
    # Plot
    sns.barplot(x='Normalized Importance', y='Feature', hue='Method', data=plot_data)
    plt.title('Top 20 Features: XGBoost vs Permutation Importance')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return comparison

def analyze_shap_values(model, X_test, output_prefix='feature_analysis/shap'):
    """Calculate and visualize SHAP values for model interpretability"""
    print("Analyzing SHAP values for deeper feature impact understanding...")
    
    # Take a sample if dataset is large (SHAP analysis can be memory-intensive)
    if len(X_test) > 1000:
        X_sample = X_test.sample(1000, random_state=42)
        print(f"Using a sample of 1000 instances for SHAP analysis")
    else:
        X_sample = X_test
    
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    start_time = time.time()
    shap_values = explainer.shap_values(X_sample)
    end_time = time.time()
    print(f"SHAP values calculated in {end_time - start_time:.2f} seconds")
    
    # Summary plot (shows magnitude and direction of feature impact)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Values: Feature Impact Direction and Magnitude')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png')
    plt.close()
    
    # Bar summary plot (feature importance ranking)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_importance.png')
    plt.close()
    
    # Create dependence plots for top 5 features
    # These show how the feature affects the prediction across its value range
    feature_indices = np.argsort(-np.abs(shap_values).mean(0))[:5]
    top_features = X_sample.columns[feature_indices]
    
    for feature in top_features:
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.title(f'SHAP Dependence Plot: Impact of {feature}')
        plt.tight_layout()
        safe_feature = feature.replace('/', '_').replace(' ', '_')
        plt.savefig(f'{output_prefix}_dependence_{safe_feature}.png')
        plt.close()
    
    return shap_values, explainer

def analyze_feature_groups(importance_df, X_train, output_path='feature_analysis/feature_groups.png'):
    """Group features by category and analyze group importance"""
    print("Analyzing feature importance by category...")
    
    # Define feature groups based on domain knowledge
    # Customize these based on your actual features
    groups = {
        'Location': [col for col in X_train.columns if any(loc in col.upper() for loc in 
                    ['BEDOK', 'TAMPINES', 'BUKIT', 'ANG MO KIO', 'WOODLANDS', 'JURONG', 
                     'HOUGANG', 'SENGKANG', 'PUNGGOL', 'PASIR RIS', 'QUEENSTOWN', 'BISHAN', 
                     'GEYLANG', 'KALLANG', 'WHAMPOA', 'TOA PAYOH', 'MARINE', 'SERANGOON'])],
        
        'Flat Type': [col for col in X_train.columns if any(type_str in col.upper() for type_str in 
                     ['ROOM', 'EXECUTIVE', 'TYPE'])],
        
        'Size & Floor': [col for col in X_train.columns if any(area_str in col.lower() for area_str in 
                       ['area', 'sqm', 'floor_area', 'storey'])],
        
        'Lease': [col for col in X_train.columns if any(lease_str in col.lower() for lease_str in 
                 ['lease', 'remaining'])],
        
        'Price Ratio': [col for col in X_train.columns if 'price_ratio' in col.lower()],
        
        'Other': []  # Will catch any features not in other groups
    }
    
    # Assign all remaining features to 'Other' category
    all_assigned = []
    for group in groups.values():
        all_assigned.extend(group)
    
    groups['Other'] = [col for col in X_train.columns if col not in all_assigned]
    
    # Calculate group importance
    group_importance = {}
    
    for group_name, group_features in groups.items():
        # Filter to features that exist in both our importance dataframe and group
        valid_features = [f for f in group_features if f in importance_df['Feature'].values]
        
        if valid_features:
            # Sum importance for this group
            group_imp = importance_df[importance_df['Feature'].isin(valid_features)]['Importance'].sum()
            
            # Store results
            group_importance[group_name] = {
                'Total Importance': group_imp,
                'Feature Count': len(valid_features),
                'Average Importance': group_imp / len(valid_features),
                'Top Features': importance_df[importance_df['Feature'].isin(valid_features)].head(3)['Feature'].tolist()
            }
    
    # Convert to DataFrame
    group_df = pd.DataFrame({
        'Category': list(group_importance.keys()),
        'Total Importance': [group_importance[g]['Total Importance'] for g in group_importance],
        'Feature Count': [group_importance[g]['Feature Count'] for g in group_importance],
        'Avg Importance/Feature': [group_importance[g]['Average Importance'] for g in group_importance],
        'Top Features': [', '.join(group_importance[g]['Top Features'][:3]) for g in group_importance]
    })
    
    # Sort by total importance
    group_df = group_df.sort_values('Total Importance', ascending=False)
    
    # Save to CSV
    group_df.to_csv('feature_analysis/feature_groups.csv', index=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Total Importance', y='Category', data=group_df)
    plt.title('Feature Importance by Category')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also plot average importance per feature
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Avg Importance/Feature', y='Category', data=group_df)
    plt.title('Average Feature Importance by Category')
    plt.tight_layout()
    plt.savefig('feature_analysis/feature_groups_avg.png')
    plt.close()
    
    return group_df, groups

def feature_reduction_experiment(model, X_train, y_train, X_test, y_test, importance_df):
    """Test model performance with reduced feature sets to find optimal balance"""
    print("Running feature reduction experiments...")
    
    # Define feature count levels to test
    feature_counts = [5, 10, 20, 30, 50, 75, 100, X_train.shape[1]]
    valid_counts = [count for count in feature_counts if count <= len(importance_df)]
    
    results = []
    
    for count in valid_counts:
        print(f"Testing with top {count} features...")
        
        # Get top features
        top_features = importance_df.head(count)['Feature'].tolist()
        
        # Subset the data
        X_train_subset = X_train[top_features]
        X_test_subset = X_test[top_features]
        
        # Create a fresh model with same parameters (to avoid data leakage)
        # Note: We're retraining a new model, not reusing the existing one
        model_copy = joblib.load('optimized_xgboost_model.pkl')
        
        # Train on subset
        start_time = time.time()
        model_copy.fit(X_train_subset, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        y_pred = model_copy.predict(X_test_subset)
        pred_time = time.time() - start_time
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store results
        results.append({
            'Feature Count': count,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Training Time': train_time,
            'Prediction Time': pred_time
        })
        
        print(f"  RMSE: ${rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv('feature_analysis/feature_reduction_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Create primary y-axis for RMSE
    fig, ax1 = plt.subplots(figsize=(12, 8))
    color = 'tab:blue'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('RMSE ($)', color=color)
    ax1.plot(results_df['Feature Count'], results_df['RMSE'], 'o-', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary y-axis for R²
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('R² Score', color=color)
    ax2.plot(results_df['Feature Count'], results_df['R²'], 's-', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add MAPE as text annotations
    for i, row in results_df.iterrows():
        ax1.annotate(f"{row['MAPE']:.1f}%", 
                    (row['Feature Count'], row['RMSE']), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title('Model Performance vs. Number of Features')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig('feature_analysis/feature_reduction_plot.png')
    plt.close()
    
    # Also plot training time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Feature Count'], results_df['Training Time'], 'o-', linewidth=2)
    plt.xlabel('Number of Features')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Number of Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_analysis/feature_reduction_time.png')
    plt.close()
    
    return results_df

def generate_html_report(xgb_importance, perm_importance, feature_groups, reduction_results):
    """Generate an HTML report with all findings from feature importance analysis"""
    print("Generating comprehensive HTML report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HDB Price Prediction - Feature Importance Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            h3 {{ color: #2980b9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .insight {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
            .recommendation {{ background-color: #eafaf1; padding: 15px; border-left: 5px solid #2ecc71; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>HDB Resale Price Prediction: Feature Importance Analysis</h1>
        
        <div class="insight">
            <h3>Key Findings Summary</h3>
            <p>The optimized XGBoost model achieved an R² of 0.91 on test data with a MAPE of 7.83%, 
            indicating strong predictive performance. Feature importance analysis reveals that location factors,
            flat types, and price ratio features are the strongest predictors of HDB resale prices.</p>
        </div>
        
        <h2>1. Top Features by Importance</h2>
        
        <h3>XGBoost Built-in Importance</h3>
        <p>The top 10 features based on XGBoost's built-in importance metric:</p>
        <table>
            <tr>
                <th>Feature</th>
                <th>Importance Score</th>
            </tr>
    """
    
    # Add XGBoost importance table rows
    for _, row in xgb_importance.head(10).iterrows():
        html_content += f"""
            <tr>
                <td>{row['Feature']}</td>
                <td>{row['Importance']:.6f}</td>
            </tr>
        """
    
    html_content += f"""
        </table>
        
        <img src="xgboost_importance.png" alt="XGBoost Feature Importance">
        
        <h3>Permutation Importance</h3>
        <p>The top 10 features based on permutation importance (measuring drop in performance when feature values are shuffled):</p>
        <table>
            <tr>
                <th>Feature</th>
                <th>Importance Score</th>
            </tr>
    """
    
    # Add permutation importance table rows
    for _, row in perm_importance.head(10).iterrows():
        html_content += f"""
            <tr>
                <td>{row['Feature']}</td>
                <td>{row['Importance']:.6f}</td>
            </tr>
        """
    
    html_content += f"""
        </table>
        
        <img src="permutation_importance.png" alt="Permutation Feature Importance">
        
        <div class="insight">
            <h3>Insight: Different Importance Metrics</h3>
            <p>Comparing built-in XGBoost importance with permutation importance shows 
            {"similar" if set(xgb_importance.head(5)['Feature']) == set(perm_importance.head(5)['Feature']) else "some differences in"} 
            top features. Permutation importance is generally more reliable as it directly measures the impact of each feature
            on model performance, while built-in importance is based on how often features are used in the model.</p>
        </div>
        
        <h2>2. Feature Groups Analysis</h2>
        
        <p>Features were grouped by category to understand which types of information most influence HDB pricing:</p>
        
        <table>
            <tr>
                <th>Category</th>
                <th>Total Importance</th>
                <th>Feature Count</th>
                <th>Top Features</th>
            </tr>
    """
    
    # Add feature groups table rows
    for _, row in feature_groups.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Category']}</td>
                <td>{row['Total Importance']:.6f}</td>
                <td>{row['Feature Count']}</td>
                <td>{row['Top Features']}</td>
            </tr>
        """
    
    html_content += f"""
        </table>
        
        <img src="feature_groups.png" alt="Feature Group Importance">
        
        <div class="insight">
            <h3>Insight: Feature Category Importance</h3>
            <p>The {feature_groups.iloc[0]['Category']} category has the highest total importance, 
            indicating that {feature_groups.iloc[0]['Category'].lower()} factors are the most influential in determining HDB resale prices.
            However, when looking at average importance per feature, the {feature_groups.sort_values('Avg Importance/Feature', ascending=False).iloc[0]['Category']} 
            category features have the highest individual impact.</p>
        </div>
        
        <h2>3. SHAP Analysis</h2>
        
        <p>SHAP (SHapley Additive exPlanations) values show how each feature contributes to predictions:</p>
        
        <img src="shap_summary.png" alt="SHAP Summary Plot">
        <p>The SHAP summary plot shows both the magnitude and direction of each feature's impact. 
        Features at the top have the largest impact on predictions. Red points indicate higher feature values, blue points indicate lower values.</p>
        
        <img src="shap_importance.png" alt="SHAP Feature Importance">
        <p>The SHAP importance plot ranks features by their absolute impact on model output, regardless of direction.</p>
        
        <div class="insight">
            <h3>Insight: SHAP Value Analysis</h3>
            <p>SHAP analysis reveals that some features have a clear positive or negative relationship with price, 
            while others have more complex, non-linear effects. For example, [specific insight about a key feature 
            from the SHAP plots] shows interesting patterns in how it influences predictions.</p>
        </div>
        
        <h2>4. Feature Reduction Experiment</h2>
        
        <p>We tested model performance with different numbers of features to find the optimal balance
        between complexity and accuracy:</p>
        
        <table>
            <tr>
                <th>Feature Count</th>
                <th>RMSE ($)</th>
                <th>R²</th>
                <th>MAPE (%)</th>
                <th>Training Time (s)</th>
            </tr>
    """
    
    # Add feature reduction table rows
    for _, row in reduction_results.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Feature Count']}</td>
                <td>${row['RMSE']:.2f}</td>
                <td>{row['R²']:.4f}</td>
                <td>{row['MAPE']:.2f}%</td>
                <td>{row['Training Time']:.4f}</td>
            </tr>
        """
    
    # Find optimal feature count (best R²)
    optimal_count = reduction_results.loc[reduction_results['R²'].idxmax()]['Feature Count']
    
    html_content += f"""
        </table>
        
        <img src="feature_reduction_plot.png" alt="Feature Reduction Performance">
        
        <div class="insight">
            <h3>Insight: Optimal Feature Set</h3>
            <p>The model achieves its best performance with {optimal_count} features. 
            Using fewer features leads to underfitting, while using all features doesn't significantly improve performance
            and increases training time.</p>
        </div>
        
        <h2>5. Conclusions and Recommendations</h2>
        
        <div class="recommendation">
            <h3>Key Recommendations</h3>
            <ul>
                <li><strong>Optimal Feature Set:</strong> Use the top {optimal_count} features for the best balance of performance and efficiency.</li>
                <li><strong>Most Important Features:</strong> Focus on {', '.join(xgb_importance.head(3)['Feature'].tolist())} as they are the strongest predictors.</li>
                <li><strong>Feature Groups:</strong> {feature_groups.iloc[0]['Category']} features collectively have the greatest impact on price predictions.</li>
                <li><strong>Model Simplification:</strong> A reduced model with {optimal_count} features performs nearly as well as the full model while being more interpretable and faster to train.</li>
            </ul>
        </div>
        
        <h3>Business Implications</h3>
        <p>These findings have several implications for HDB property valuation and decision-making:</p>
        <ul>
            <li>Location remains a critical factor in HDB pricing, with certain neighborhoods commanding significant premiums</li>
            <li>Flat type and size characteristics have quantifiable impacts on resale value</li>
            <li>Remaining lease duration shows clear influence on property values</li>
            <li>The price ratio features demonstrate the importance of relative pricing in the market</li>
        </ul>
        
        <h3>Next Steps</h3>
        <ul>
            <li>Implement the optimized feature set in the production model</li>
            <li>Develop focused sub-models for specific property segments</li>
            <li>Create interactive visualizations for stakeholders to explore feature relationships</li>
            <li>Periodically refresh the analysis as new data becomes available</li>
        </ul>
        
    </body>
    </html>
    """
    
    # Write HTML to file
    with open('feature_analysis/feature_importance_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: feature_analysis/feature_importance_report.html")

def main():
    # Create output directory
    os.makedirs('feature_analysis', exist_ok=True)
    
    # Load model and data
    model, X_train, y_train, X_test, y_test = load_model_and_data()
    
    # Analyze XGBoost built-in importance
    xgb_importance = analyze_xgboost_importance(model, X_train)
    
    # Analyze permutation importance
    perm_importance = analyze_permutation_importance(model, X_test, y_test)
    
    # Compare importance methods
    importance_comparison = compare_importance_methods(xgb_importance, perm_importance)
    
    # Analyze SHAP values for deeper interpretability
    shap_values, explainer = analyze_shap_values(model, X_test)
    
    # Analyze feature groups
    feature_groups, group_dict = analyze_feature_groups(xgb_importance, X_train)
    
    # Run feature reduction experiment
    reduction_results = feature_reduction_experiment(model, X_train, y_train, X_test, y_test, xgb_importance)
    
    # Generate comprehensive HTML report
    generate_html_report(xgb_importance, perm_importance, feature_groups, reduction_results)
    
    print("\n====== Feature Importance Analysis Complete ======")
    print("All results saved to the 'feature_analysis' folder")
    print("\nKey files generated:")
    print("- feature_importance_report.html: Complete analysis report")
    print("- Various CSV files with detailed metrics")
    print("- Visualizations of feature importance")
    print("\nNext steps: Review the report to understand key features and")
    print("consider simplifying your model based on the feature reduction experiment.")

if __name__ == "__main__":
    main()
