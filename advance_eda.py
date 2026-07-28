import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_by_ai(df):
    """
    Performs a comprehensive, end-to-end Advanced Exploratory Data Analysis (EDA) 
    on the provided pandas DataFrame `df`.
    """
    # Set visual styles for plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.autolayout': True})

    print("=" * 80)
    print("1. SETUP & INITIALIZATION COMPLETED")
    print("=" * 80)

    # =========================================================================
    # 2. DATASET OVERVIEW & DESCRIPTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. DATASET OVERVIEW & DESCRIPTION")
    print("=" * 80)

    print("\n--- Shape of the Dataset ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Data Types & Non-Null Counts ---")
    print(df.info())

    print("\n--- Statistical Summary (Numerical Columns) ---")
    print(df.describe())

    print("\n--- Statistical Summary (Categorical/Object Columns) ---")
    cat_summary = df.describe(include=['O', 'category'])
    if not cat_summary.empty:
        print(cat_summary)
    else:
        print("No categorical/object columns found.")

    print("\n--- Missing Values Count ---")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0] if missing_vals.sum() > 0 else "No missing values found.")

    print("\n--- Duplicate Rows Count ---")
    dup_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {dup_count}")

    # =========================================================================
    # 3. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. CORRELATION ANALYSIS")
    print("=" * 80)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) > 1:
        print("\nCalculating correlation matrix for numerical columns...")
        corr_matrix = df[num_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=True)
        plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold')
        plt.show()
    else:
        print("Not enough numerical columns to compute a meaningful correlation matrix.")

    # =========================================================================
    # 4. UNIVARIATE NUMERICAL COLUMN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. UNIVARIATE NUMERICAL COLUMN ANALYSIS")
    print("=" * 80)

    if num_cols:
        print("\nCalculating skewness and plotting distributions for numerical columns...")
        for col in num_cols:
            series = df[col].dropna()
            skew_val = series.skew()
            print(f"Column: '{col}' | Skewness: {skew_val:.4f}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histplot with KDE
            sns.histplot(series, kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f'Histogram & KDE: {col}')
            axes[0].set_xlabel(col)

            # Boxplot
            sns.boxplot(x=series, ax=axes[1], color='lightgreen')
            axes[1].set_title(f'Boxplot (Outliers): {col}')
            axes[1].set_xlabel(col)

            plt.suptitle(f'Univariate Analysis of {col}', fontsize=12, fontweight='bold', y=1.03)
            plt.show()
    else:
        print("No numerical columns available for analysis.")

    # =========================================================================
    # 5. UNIVARIATE OBJECT (CATEGORICAL) COLUMN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. UNIVARIATE OBJECT (CATEGORICAL) COLUMN ANALYSIS")
    print("=" * 80)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_cols:
        for col in cat_cols:
            unique_count = df[col].nunique()
            print(f"\nAnalyzing categorical column: '{col}' (Unique values: {unique_count})")

            if unique_count < 20:
                plt.figure(figsize=(10, 5))
                sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
                plt.title(f'Count Plot of {col}', fontsize=12, fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.show()
            else:
                print(f"High-cardinality column detected. Displaying top 10 most frequent categories for '{col}':")
                top_10 = df[col].value_counts().head(10)
                print(top_10)
                
                plt.figure(figsize=(10, 5))
                sns.barplot(x=top_10.index, y=top_10.values, palette='viridis')
                plt.title(f'Top 10 Categories of {col}', fontsize=12, fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.show()
    else:
        print("No categorical/object columns available for analysis.")

    # =========================================================================
    # 6. BIVARIATE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. BIVARIATE ANALYSIS")
    print("=" * 80)

    # Numerical vs Categorical (Box/Violin plot)
    if num_cols and cat_cols:
        target_num = num_cols[0]
        target_cat = cat_cols[0]
        print(f"\nGenerating Boxplot of Numerical '{target_num}' across Categorical '{target_cat}'...")
        
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=target_cat, y=target_num, palette='Set3')
        plt.title(f'{target_num} distribution across {target_cat}', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    # Numerical vs Numerical (Scatter plot with regression line)
    if len(num_cols) >= 2:
        num_x, num_y = num_cols[0], num_cols[1]
        print(f"\nGenerating Regression Scatter Plot between '{num_x}' and '{num_y}'...")
        
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x=num_x, y=num_y, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.title(f'Regression Plot: {num_x} vs {num_y}', fontsize=12, fontweight='bold')
        plt.show()
    else:
        print("Not enough numerical columns for scatter/regression bivariate analysis.")

    # =========================================================================
    # 7. TIME SERIES ANALYSIS (CONDITIONAL)
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. TIME SERIES ANALYSIS (CONDITIONAL)")
    print("=" * 80)

    date_col = None
    # Check explicitly for datetime types or potential datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        elif df[col].dtype == 'object':
            # Try parsing a small sample to see if it fits date formats
            try:
                pd.to_datetime(df[col].dropna().iloc[:5])
                date_col = col
                break
            except (ValueError, TypeError, IndexError):
                continue

    if date_col:
        print(f"\nDate/Time column identified: '{date_col}'. Proceeding with Time Series Analysis...")
        ts_df = df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.sort_values(by=date_col)
        
        # Set date column as index
        ts_df.set_index(date_col, inplace=True)
        
        # Pick a numerical metric for resampling, default to the first numerical column if available
        metric_col = num_cols[0] if num_cols else None
        
        if metric_col:
            print(f"Resampling and plotting monthly average trend for '{metric_col}'...")
            resampled_data = ts_df[metric_col].resample('ME').mean() # 'ME' for Month End
            
            plt.figure(figsize=(12, 6))
            plt.plot(resampled_data.index, resampled_data.values, marker='o', linestyle='-', label='Monthly Mean')
            
            # Optional: Rolling average (e.g., 3 periods moving average)
            rolling_avg = resampled_data.rolling(window=3, min_periods=1).mean()
            plt.plot(rolling_avg.index, rolling_avg.values, color='orange', linestyle='--', label='3-Month Rolling Avg')
            
            plt.title(f'Time Series Trend for {metric_col}', fontsize=12, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel(metric_col)
            plt.legend()
            plt.show()
        else:
            print("No numerical column available to aggregate for Time Series analysis.")
    else:
        print("No date/time column found in the dataset. Skipping Time Series Analysis.")

    # =========================================================================
    # 8. MULTIVARIATE ANALYSIS (CATEGORICAL GROUPING WITH HUE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. MULTIVARIATE ANALYSIS")
    print("=" * 80)

    # Check for specific or fallback columns matching Sales, Region, Segment requirements
    lower_cols = [c.lower() for c in df.columns]
    
    def find_matching_col(keywords):
        for kw in keywords:
            for i, c in enumerate(lower_cols):
                if kw in c:
                    return df.columns[i]
        return None

    mv_sales = find_matching_col(['sales', 'revenue', 'amount', 'total', 'price'])
    mv_region = find_matching_col(['region', 'country', 'state', 'location', 'zone'])
    mv_segment = find_matching_col(['segment', 'category', 'type', 'class', 'group'])

    # Fallback to general numerical/categorical columns if exact semantic matches aren't found
    if not mv_sales and num_cols:
        mv_sales = num_cols[0]
    if not mv_region and len(cat_cols) >= 1:
        mv_region = cat_cols[0]
    if not mv_segment and len(cat_cols) >= 2:
        mv_segment = cat_cols[1]

    if mv_sales and mv_region and mv_segment and mv_region != mv_segment:
        print(f"\nGenerating Grouped Bar Plot using:")
        print(f" - Numerical (Y-axis): {mv_sales}")
        print(f" - Categorical 1 (X-axis): {mv_region}")
        print(f" - Categorical 2 (Hue): {mv_segment}")

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df, 
            x=mv_region, 
            y=mv_sales, 
            hue=mv_segment, 
            estimator=np.mean, 
            palette='viridis', 
            errorbar=None
        )
        plt.title(f'Grouped Bar Plot: {mv_sales} by {mv_region} and {mv_segment}', fontsize=12, fontweight='bold')
        plt.xlabel(mv_region)
        plt.ylabel(f'Mean of {mv_sales}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=mv_segment, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Annotate values on bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{height:.1f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=8, color='black',
                            xytext=(0, 3),
                            textcoords='offset points')
                            
        plt.show()
    else:
        print("Skipping grouped bar plot: Insufficient valid column combinations found for multivariate interaction.")

    # Pairplot / FacetGrid illustration if enough numerical columns exist
    if len(num_cols) >= 3 and cat_cols:
        print("\nGenerating Pairplot colored by category...")
        subset_cols = num_cols[:3] + [cat_cols[0]]
        sns.pairplot(df[subset_cols].dropna(), hue=cat_cols[0], palette='Set1')
        plt.show()

    print("\n" + "=" * 80)
    print("EDA PIPELINE COMPLETE")
    print("=" * 80)
