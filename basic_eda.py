import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame):
    """
    Performs comprehensive basic Exploratory Data Analysis (EDA) on a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    """
    print("=" * 60)
    print(" 📊 BASIC EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)
    
    # 1. Dataset Shape
    print("\n[1. DATASET SHAPE]")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    
    # 2. Data Types & Non-Null Counts
    print("\n[2. COLUMN DATA TYPES & NON-NULL COUNTS]")
    print("-" * 40)
    df.info()
    
    # 3. Missing Values Summary
    print("\n[3. MISSING VALUES SUMMARY]")
    print("-" * 40)
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if not missing_df.empty:
        print(missing_df.sort_values(by='Missing Count', ascending=False))
    else:
        print("🎉 Good news! There are no missing values in this dataset.")
        
    # 4. Duplicate Rows
    print("\n[4. DUPLICATE ROWS]")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates} ({(duplicates / len(df)) * 100:.2f}% of dataset)")
    
    # 5. Numerical Statistics & Skewness
    print("\n[5. NUMERICAL FEATURES SUMMARY]")
    print("-" * 40)
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe().T
        desc['skewness'] = num_df.skew()
        print(desc[['count', 'mean', 'std', 'min', '50%', 'max', 'skewness']])
    else:
        print("No numerical columns found in the dataset.")
        
    # 6. Categorical Statistics
    print("\n[6. CATEGORICAL FEATURES SUMMARY]")
    print("-" * 40)
    cat_df = df.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        cat_summary = []
        for col in cat_df.columns:
            cat_summary.append({
                'Column': col,
                'Unique Values': cat_df[col].nunique(),
                'Top Value': cat_df[col].mode()[0] if not cat_df[col].mode().empty else np.nan,
                'Top Frequency': cat_df[col].value_counts().iloc[0] if not cat_df[col].empty else 0
            })
        print(pd.DataFrame(cat_summary).to_string(index=False))
    else:
        print("No categorical columns found in the dataset.")
        
    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
