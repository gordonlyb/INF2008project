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
        df = df.drop(columns=['month'])

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
    input_file = "Dataset/Resaleflatprices.csv"
    output_file = "Dataset/processed_Resaleflatprices.csv"

    print("Processing dataset...")
    processed_df = load_and_process(input_file, output_file)

    if processed_df is not None:
        print("Sample output:")
        print(processed_df.head())
    else:
        print("Processing failed.")