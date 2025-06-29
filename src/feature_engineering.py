# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Perform feature engineering on raw transaction data, including time-based and aggregate features,
#          and calculate RFM metrics.
# Key features:
# - Loads data from an Excel file.
# - Extracts time-based features from transaction timestamps.
# - Creates aggregate features at the customer level.
# - Calculates Recency, Frequency, and Monetary (RFM) features.
# - Preprocesses data using scikit-learn pipelines for numerical scaling and categorical one-hot encoding.
# - Saves key summary reports to CSV files in a 'reports/' directory.
# - Generates and saves plots visualizing engineered features and RFM distributions in a 'plots/' directory.
# Task Number: 3


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os # Import the os module for directory operations
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced plots


class FeatureEngineering:
    """
    A class to perform feature engineering on transaction data,
    including data loading, feature extraction, aggregation,
    RFM calculation, preprocessing, and reporting/plotting.
    """

    def __init__(self):
        pass

    def load_data(self, file_path):
        """
        Load the raw data from Excel file.

        Args:
            file_path (str): The path to the Excel file.

        Returns:
            pd.DataFrame: The loaded DataFrame, or None if an error occurs.
        """
        print(f"Loading data from {file_path}...")
        try:
            df = pd.read_excel(file_path)
            print("Data loaded successfully.")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def extract_transaction_features(self, df):
        """
        Extract time-based features from transaction timestamps.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'TransactionStartTime'.

        Returns:
            pd.DataFrame: The DataFrame with added time-based features.
        """
        print("Extracting time-based transaction features...")
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("DataFrame must contain 'TransactionStartTime' column.")

        # Convert to datetime if not already
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        # Drop rows where TransactionStartTime could not be parsed
        df.dropna(subset=['TransactionStartTime'], inplace=True)

        if df.empty:
            print("Warning: DataFrame became empty after cleaning TransactionStartTime.")
            return df

        # Extract hour, day, month, year from timestamp
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year

        # Time since midnight in minutes
        df['minutes_since_midnight'] = (df['TransactionStartTime'].dt.hour * 60) + df['TransactionStartTime'].dt.minute

        # Weekend flag (Monday=0, Sunday=6)
        df['is_weekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int) # Convert boolean to int (0 or 1)

        print("Time-based transaction features extracted.")
        return df

    def create_aggregate_features(self, df):
        """
        Create aggregate features at customer level.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added aggregate customer features.
        """
        print("Creating aggregate features per customer...")
        if 'CustomerId' not in df.columns:
            raise ValueError("DataFrame must contain 'CustomerId' column for aggregation.")
        if 'Amount' not in df.columns:
            raise ValueError("DataFrame must contain 'Amount' column for aggregation.")
        if 'Value' not in df.columns:
            raise ValueError("DataFrame must contain 'Value' column for aggregation.")
        if 'ProductCategory' not in df.columns:
            print("Warning: 'ProductCategory' not found, 'unique_product_categories' will be skipped.")
            has_product_category = False
        else:
            has_product_category = True

        # Ensure 'CustomerId' is of a hashable type if it's not (e.g., if it was mixed types)
        df['CustomerId'] = df['CustomerId'].astype(str)

        # Calculate total transactions per customer
        transaction_count = df.groupby('CustomerId').size().reset_index(name='total_transactions')
        df = df.merge(transaction_count, on='CustomerId', how='left')

        # Calculate average transaction amount per customer
        avg_amount = df.groupby('CustomerId')['Amount'].mean().reset_index(name='avg_transaction_amount')
        df = df.merge(avg_amount, on='CustomerId', how='left')

        # Calculate standard deviation of transaction amounts (handle cases with single transaction)
        std_amount = df.groupby('CustomerId')['Amount'].std().reset_index(name='std_transaction_amount')
        # Fill NaN for customers with a single transaction, as std dev is undefined
        std_amount['std_transaction_amount'].fillna(0, inplace=True)
        df = df.merge(std_amount, on='CustomerId', how='left')

        # Calculate total transaction value per customer
        total_value = df.groupby('CustomerId')['Value'].sum().reset_index(name='total_transaction_value')
        df = df.merge(total_value, on='CustomerId', how='left')

        # Calculate number of unique product categories per customer
        if has_product_category:
            unique_categories = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index(name='unique_product_categories')
            df = df.merge(unique_categories, on='CustomerId', how='left')
        else:
            df['unique_product_categories'] = 0 # Default to 0 if column is missing

        print("Aggregate features created.")
        return df

    def create_rfm_features(self, df, snapshot_date=None):
        """
        Create RFM (Recency, Frequency, Monetary) features.
        Recency: Days since last transaction.
        Frequency: Total number of transactions.
        Monetary: Total sum of 'Value' for all transactions.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'TransactionStartTime', 'TransactionId', 'Value', 'CustomerId'.
            snapshot_date (datetime): The date to calculate recency against. If None, uses max transaction date + 1 day.

        Returns:
            pd.DataFrame: A DataFrame with 'CustomerId', 'recency', 'frequency', 'monetary' columns.
        """
        print("Creating RFM features...")
        required_cols_for_rfm = ['TransactionStartTime', 'TransactionId', 'Value', 'CustomerId']
        if not all(col in df.columns for col in required_cols_for_rfm):
            raise ValueError(f"DataFrame must contain '{required_cols_for_rfm}' for RFM calculation.")

        # Ensure 'TransactionStartTime' is datetime and handle NaT
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df_rfm_valid = df.dropna(subset=['TransactionStartTime', 'Value', 'CustomerId']).copy()

        if df_rfm_valid.empty:
            print("Warning: DataFrame became empty for RFM calculation after dropping NaNs.")
            return pd.DataFrame(columns=['CustomerId', 'recency', 'frequency', 'monetary'])

        if snapshot_date is None:
            snapshot_date = df_rfm_valid['TransactionStartTime'].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(snapshot_date) # Ensure snapshot_date is datetime

        rfm_df = df_rfm_valid.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum')
        )

        rfm_df.rename(columns={
            'Recency': 'recency',
            'Frequency': 'frequency',
            'Monetary': 'monetary'
        }, inplace=True)

        print("RFM features created.")
        return rfm_df.reset_index()

    def preprocess_data(self, df):
        """
        Preprocess data with appropriate transformations (imputation, scaling, one-hot encoding).
        This prepares the data for model training.

        Args:
            df (pd.DataFrame): The DataFrame with all engineered features.

        Returns:
            tuple:
                - X (np.ndarray): The transformed feature matrix.
                - preprocessor (sklearn.compose.ColumnTransformer): The fitted preprocessor.
        """
        print("Preprocessing data for model training...")
        # Define numeric and categorical columns for preprocessing
        # These should be the columns that will be fed into the model
        numeric_features = [
            'Amount', 'Value', 'total_transactions', 'avg_transaction_amount',
            'std_transaction_amount', 'total_transaction_value',
            'unique_product_categories', 'recency', 'frequency', 'monetary',
            'transaction_hour', 'transaction_day', 'transaction_month',
            'transaction_year', 'minutes_since_midnight', 'is_weekend'
        ]

        categorical_features = [
            'ChannelId', 'ProviderId', 'ProductCategory', 'PricingStrategy', 'CountryCode'
        ]

        # Filter features to only include those present in the current DataFrame
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        # Create preprocessing pipeline for numerical features
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # Impute NaNs with median for numerical
            ('scaler', StandardScaler()) # Scale numerical features
        ])

        # Create preprocessing pipeline for categorical features
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Impute NaNs with 'missing' category
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # Drop columns not specified in transformers
        )

        # Apply transformations
        X = preprocessor.fit_transform(df)
        print("Data preprocessing complete.")
        return X, preprocessor

    def save_to_csv(self, df, filename, output_dir='reports/'):
        """
        Saves a DataFrame or Series to a CSV file.

        Args:
            df (pd.DataFrame or pd.Series): The DataFrame or Series to save.
            filename (str): The name of the CSV file (e.g., 'summary_stats.csv').
            output_dir (str): The directory to save the CSV file. Defaults to 'reports/'.
        """
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        filepath = os.path.join(output_dir, filename)
        if isinstance(df, pd.Series):
            df.name = df.name if df.name else 'value' # Ensure series has a name for CSV header
            df.to_csv(filepath, index=True, header=True) # index=True for series (e.g., describe() output)
        else:
            df.to_csv(filepath, index=False)
        print(f"Saved {filename} to {filepath}")

    def save_plot(self, fig, filename, output_dir='plots/'):
        """
        Saves a matplotlib figure to a file.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            filename (str): The name of the plot file (e.g., 'rfm_distributions.png').
            output_dir (str): The directory to save the plot. Defaults to 'plots/'.
        """
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight') # bbox_inches='tight' prevents labels/titles from being cut off
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved plot {filename} to {filepath}")

    def generate_summary_reports(self, df, rfm_df, output_dir='reports/'):
        """
        Generates and saves summary reports of the engineered features and RFM metrics.

        Args:
            df (pd.DataFrame): The main DataFrame with engineered features.
            rfm_df (pd.DataFrame): The RFM DataFrame.
            output_dir (str): Directory to save the CSV reports.
        """
        print("Generating summary reports...")

        # Overall descriptive statistics for numerical features
        numerical_df_describe = df.select_dtypes(include=np.number).describe()
        self.save_to_csv(numerical_df_describe, 'numerical_features_summary.csv', output_dir)

        # RFM descriptive statistics
        if not rfm_df.empty:
            rfm_describe = rfm_df[['recency', 'frequency', 'monetary']].describe()
            self.save_to_csv(rfm_describe, 'rfm_summary.csv', output_dir)
        else:
            print("Warning: RFM DataFrame is empty, skipping RFM summary report.")

        # Value counts for key categorical features (top N)
        categorical_cols = ['CountryCode', 'ProductCategory', 'ChannelId', 'ProviderId']
        for col in categorical_cols:
            if col in df.columns:
                # Get value counts, handle NaNs, and save top 20 for brevity
                value_counts = df[col].value_counts(dropna=False).head(20).reset_index()
                value_counts.columns = [col, 'count']
                self.save_to_csv(value_counts, f'{col}_value_counts.csv', output_dir)
            else:
                print(f"Warning: Categorical column '{col}' not found in DataFrame, skipping value counts report.")

        print("Summary reports generated.")

    def plot_engineered_features(self, df, rfm_df, output_dir='plots/'):
        """
        Generates and saves plots visualizing engineered features and RFM distributions.

        Args:
            df (pd.DataFrame): The main DataFrame with engineered features.
            rfm_df (pd.DataFrame): The RFM DataFrame.
            output_dir (str): Directory to save the plots.
        """
        print("Generating plots for engineered features...")

        # Plot RFM distributions
        if not rfm_df.empty:
            fig_rfm, axes_rfm = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(rfm_df['recency'], kde=True, ax=axes_rfm[0], color='skyblue')
            axes_rfm[0].set_title('Recency Distribution')
            axes_rfm[0].set_xlabel('Recency (Days)')
            axes_rfm[0].set_ylabel('Number of Customers')

            sns.histplot(rfm_df['frequency'], kde=True, ax=axes_rfm[1], color='lightcoral')
            axes_rfm[1].set_title('Frequency Distribution')
            axes_rfm[1].set_xlabel('Frequency (Transactions)')
            axes_rfm[1].set_ylabel('Number of Customers')

            sns.histplot(rfm_df['monetary'], kde=True, ax=axes_rfm[2], color='lightgreen')
            axes_rfm[2].set_title('Monetary Distribution')
            axes_rfm[2].set_xlabel('Monetary (Total Spend)')
            axes_rfm[2].set_ylabel('Number of Customers')
            plt.tight_layout()
            self.save_plot(fig_rfm, 'rfm_distributions.png', output_dir)
        else:
            print("Warning: RFM DataFrame is empty, skipping RFM distribution plots.")

        # Plot distributions of key aggregate features
        aggregate_features = ['total_transactions', 'avg_transaction_amount', 'total_transaction_value']
        for feature in aggregate_features:
            if feature in df.columns:
                fig_agg, ax_agg = plt.subplots(figsize=(8, 5))
                sns.histplot(df[feature].dropna(), kde=True, ax=ax_agg, color='purple')
                ax_agg.set_title(f'Distribution of {feature.replace("_", " ").title()}')
                ax_agg.set_xlabel(feature.replace("_", " ").title())
                ax_agg.set_ylabel('Number of Transactions/Customers')
                self.save_plot(fig_agg, f'{feature}_distribution.png', output_dir)
            else:
                print(f"Warning: Aggregate feature '{feature}' not found in DataFrame, skipping plot.")

        # Plot transaction patterns by hour and month
        if 'transaction_hour' in df.columns and 'transaction_month' in df.columns:
            fig_time, axes_time = plt.subplots(1, 2, figsize=(16, 5))

            sns.countplot(x='transaction_hour', data=df, palette='viridis', ax=axes_time[0])
            axes_time[0].set_title('Transaction Count by Hour of Day')
            axes_time[0].set_xlabel('Hour of Day')
            axes_time[0].set_ylabel('Transaction Count')

            sns.countplot(x='transaction_month', data=df, palette='magma', ax=axes_time[1])
            axes_time[1].set_title('Transaction Count by Month')
            axes_time[1].set_xlabel('Month')
            axes_time[1].set_ylabel('Transaction Count')
            plt.tight_layout()
            self.save_plot(fig_time, 'transaction_time_patterns.png', output_dir)
        else:
            print("Warning: Time-based features (transaction_hour or transaction_month) not found, skipping time patterns plot.")

        # Plot count of top N Product Categories
        if 'ProductCategory' in df.columns:
            fig_prod, ax_prod = plt.subplots(figsize=(10, 6))
            df['ProductCategory'].value_counts().head(10).plot(kind='bar', ax=ax_prod, color='teal')
            ax_prod.set_title('Top 10 Product Categories')
            ax_prod.set_xlabel('Product Category')
            ax_prod.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            self.save_plot(fig_prod, 'top_product_categories.png', output_dir)
        else:
            print("Warning: 'ProductCategory' not found, skipping top product categories plot.")


        print("Plots for engineered features generated.")


    def process_data(self, raw_data_path):
        """
        Complete pipeline for processing data, including loading, feature extraction,
        aggregation, RFM calculation, and saving reports/plots.

        Args:
            raw_data_path (str): The path to the raw data file (e.g., Excel).

        Returns:
            tuple:
                - df (pd.DataFrame): The full DataFrame with all engineered features.
                - rfm_df (pd.DataFrame): The DataFrame containing RFM features.
        """
        print("Starting full feature engineering pipeline (Task 3).")

        # Load data
        df = self.load_data(raw_data_path)
        if df is None:
            return None, None

        # Extract transaction features
        df = self.extract_transaction_features(df.copy()) # Use copy to avoid modifying original df

        # Create aggregate features
        df = self.create_aggregate_features(df.copy())

        # Create RFM features
        rfm_df = self.create_rfm_features(df.copy())
        # Merge RFM back to the main dataframe for a complete view
        df = df.merge(rfm_df, on='CustomerId', how='left', suffixes=('', '_rfm'))

        # Generate and save summary reports
        self.generate_summary_reports(df.copy(), rfm_df.copy())

        # Generate and save plots
        self.plot_engineered_features(df.copy(), rfm_df.copy())

        print("Full feature engineering pipeline complete.")
        return df, rfm_df

# Example Usage (demonstrates the FeatureEngineering class)
if __name__ == "__main__":
    # Create a dummy Excel file for demonstration
    dummy_data = {
        'CustomerId': [1, 1, 2, 2, 3, 4, 1, 5, 5, 6, 7, 7, 8, 8, 9, 10],
        'TransactionId': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
        'TransactionStartTime': pd.to_datetime([
            '2024-01-01 10:30:00', '2024-01-15 11:00:00', '2024-02-01 14:15:00',
            '2024-02-10 09:00:00', '2024-03-01 18:45:00', '2024-03-10 22:00:00',
            '2024-01-05 13:00:00', '2024-04-01 16:30:00', '2024-04-05 10:00:00',
            '2024-05-01 08:00:00', '2024-05-05 11:30:00', '2024-05-06 14:00:00',
            '2024-06-01 17:00:00', '2024-06-02 09:00:00', '2024-06-15 10:00:00',
            '2024-06-20 12:00:00'
        ]),
        'Amount': [100, 50, 200, 75, 300, 10, 60, 150, 25, 400, 80, 120, 5, 15, 250, 70],
        'Value': [10, 5, 20, 7, 30, 1, 6, 15, 2, 40, 8, 12, 0.5, 1.5, 25, 7],
        'CountryCode': ['US', 'US', 'CA', 'CA', 'MX', 'US', 'US', 'DE', 'DE', 'FR', 'FR', 'FR', 'GB', 'GB', 'US', 'CA'],
        'ProviderId': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'PricingStrategy': [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'ProductCategory': ['Electronics', 'Books', 'Electronics', 'Home', 'Books',
                            'Groceries', 'Electronics', 'Books', 'Groceries', 'Home',
                            'Electronics', 'Electronics', 'Books', 'Home', 'Electronics', 'Books'],
        'ChannelId': ['Online', 'Store', 'Online', 'Store', 'Online', 'Store',
                      'Online', 'Store', 'Online', 'Store', 'Online', 'Store', 'Online', 'Store', 'Online', 'Store']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_excel_path = 'dummy_transactions.xlsx'
    dummy_df.to_excel(dummy_excel_path, index=False)
    print(f"Dummy Excel file created at: {dummy_excel_path}")


    fe_processor = FeatureEngineering()
    try:
        processed_df, rfm_features_df = fe_processor.process_data(dummy_excel_path)

        if processed_df is not None and rfm_features_df is not None:
            print("\n--- Feature Engineering Results ---")
            print("\nProcessed DataFrame Head (with all engineered features):")
            print(processed_df.head())
            print("\nRFM Features DataFrame Head:")
            print(rfm_features_df.head())
            print(f"\nAll generated reports are in the 'reports/' directory.")
            print(f"All generated plots are in the 'plots/' directory.")
        else:
            print("Feature engineering pipeline did not return valid DataFrames.")

        # Clean up the dummy Excel file
        if os.path.exists(dummy_excel_path):
            os.remove(dummy_excel_path)
            print(f"\nCleaned up dummy Excel file: {dummy_excel_path}")

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")

