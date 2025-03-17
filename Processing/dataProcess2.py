import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def drop_columns(df):
    columns_to_drop = ["lease_commence_date", "street_name", "block"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')  # 'ignore' prevents errors if column is missing
    return df

#onehotencoding to map category to binary
def town_encoding(df):
    if "town" in df.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first column to avoid redundancy
        town_encoded = encoder.fit_transform(df[['town']])

        town_names = [col.split("_")[1] for col in encoder.get_feature_names_out(['town'])]


        town_df = pd.DataFrame(town_encoded, columns=town_names, index=df.index)
        
        df = df.drop(columns=['town'])  # Drop original column
        df = pd.concat([df, town_df], axis=1)  # Merge encoded columns
    return df

def flat_encoding(df):
    if "flat_type" in df.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')  
        flat_encoded = encoder.fit_transform(df[['flat_type']])

        flat_names = [col.replace("flat_type_", "") for col in encoder.get_feature_names_out(['flat_type'])]

        flat_df = pd.DataFrame(flat_encoded, columns=flat_names, index=df.index)
        
        df = df.drop(columns=['flat_type'])
        df = pd.concat([df, flat_df], axis=1)
    return df

def flat_model_encoding(df):
    if "flat_model" in df.columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')  
        model_encoded = encoder.fit_transform(df[['flat_model']])

        model_names = [col.replace("flat_model_", "") for col in encoder.get_feature_names_out(['flat_model'])]
        model_df = pd.DataFrame(model_encoded, columns=model_names, index=df.index)
        
        df = df.drop(columns=['flat_model'])
        df = pd.concat([df, model_df], axis=1)
    return df

def convert_lease(lease_str):
    try:
        parts = lease_str.split()
        if len(parts) == 2:  # "50 years"
            years = int(parts[0])
            months = 0
        elif len(parts) >= 3:  # "74 years 08 months"
            years = int(parts[0])
            months = int(parts[2])
        else:
            raise ValueError(f"Unexpected lease format: {lease_str}")
        return years + months / 12
    except Exception as e:
        print(f"Error parsing lease: {lease_str}. Defaulting to NaN.")
        return np.nan

def lease_encoding(df):
    if "remaining_lease" in df.columns:
        df['remaining_lease'] = df['remaining_lease'].apply(convert_lease)
    return df

def storey_encoding(df):
    if "storey_range" in df.columns:
        df["storey_range"] = df["storey_range"].str.split(" TO ").apply(lambda x: (int(x[0]) + int(x[1])) / 2)
        df.rename(columns={"storey_range": "Average_storey_range"}, inplace=True)
    return df

# New Function: Domain-Specific Feature Engineering
def engineer_domain_features(df):
    """Add domain-specific features to enhance model performance"""
    df = df.copy()
    
    # Create lease decay features
    if 'remaining_lease' in df.columns:
        # Exponential decay function to capture non-linear lease value
        df['lease_exp_factor'] = np.exp(df['remaining_lease']/99) - 1
        
        # Categorical lease indicators
        df['short_lease'] = (df['remaining_lease'] <= 60).astype(int)
        df['long_lease'] = (df['remaining_lease'] >= 80).astype(int)
    
    # Create interaction features
    if 'floor_area_sqm' in df.columns and 'Average_storey_range' in df.columns:
        df['area_by_storey'] = df['floor_area_sqm'] * df['Average_storey_range']
    
    # Create price ratio features (if you have price data)
    if 'resale_price' in df.columns:
        # Calculate town-based price factors
        for col in df.columns:
            # Only process categorical columns that were one-hot encoded
            if col in ['floor_area_sqm', 'Average_storey_range', 'remaining_lease', 
                      'resale_price', 'lease_exp_factor', 'short_lease', 
                      'long_lease', 'area_by_storey']:
                continue
                
            # Check if column contains only 0s and 1s (one-hot encoded)
            if set(df[col].unique()).issubset({0, 1}):
                # Only calculate for columns with both 0s and 1s
                if df[col].nunique() > 1:
                    # Calculate average price for when feature is present (1) vs absent (0)
                    avg_price_present = df[df[col] == 1]['resale_price'].mean()
                    avg_price_overall = df['resale_price'].mean()
                    
                    # Create price ratio feature
                    if avg_price_overall > 0:  # Avoid division by zero
                        ratio_name = f'{col}_price_ratio'
                        # For rows where feature is present, use the ratio; otherwise, use 1.0
                        df[ratio_name] = 1.0
                        df.loc[df[col] == 1, ratio_name] = avg_price_present / avg_price_overall
    
    return df

def load_and_process(file_path, output_path):
    try:
        df = pd.read_csv(file_path)

        #drop original column
        df.drop(columns=['date'], inplace=True, errors='ignore')
        df = drop_columns(df)

        df = town_encoding(df)
        df = flat_encoding(df)
        df = storey_encoding(df)
        df = flat_model_encoding(df)
        df = lease_encoding(df)
        df = df.drop(columns=['month'], errors='ignore')
        
        # Add domain-specific features
        df = engineer_domain_features(df)

        scaler = MinMaxScaler()
        numerical_cols = ['floor_area_sqm', 'Average_storey_range', 'remaining_lease']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
def processedCSV():
    input_file = "../Dataset/Resaleflatprices.csv"
    output_file = "../Dataset/processed_Resaleflatprices_XGB.csv"

    print("Processing dataset...")
    processed_df = load_and_process(input_file, output_file)

    if processed_df is not None:
        print("Sample output:")
        print(processed_df.head())
        print(f"Dataset shape: {processed_df.shape}")
        print(f"Columns added: {processed_df.columns.tolist()}")
    else:
        print("Processing failed.")

# If you run this file directly
if __name__ == "__main__":
    processedCSV()
