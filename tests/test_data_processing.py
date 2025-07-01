import pandas as pd
import numpy as np
from src.data_processing import DataProcessing
from src.feature_engineering import FeatureEngineering
import pytest


def test_create_rfm_features():
    """Test the RFM feature creation function"""
    # Create sample data
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-15'),
            pd.Timestamp('2023-02-01')
        ],
        'Value': [100, 200, 50]
    })
    
    dp = DataProcessing()
    rfm_df = dp.create_rfm_features(data, snapshot_date=pd.Timestamp('2023-02-10'))
    
    assert len(rfm_df) == 2  # Two unique customers
    assert 'recency' in rfm_df.columns
    assert 'frequency' in rfm_df.columns
    assert 'monetary' in rfm_df.columns
    
    # Check recency calculation
    assert rfm_df.loc[rfm_df['CustomerId'] == 1, 'recency'].values[0] == 25  # Days from Jan 15 to Feb 9
    assert rfm_df.loc[rfm_df['CustomerId'] == 2, 'recency'].values[0] == 9  # Days from Feb 1 to Feb 9


def test_identify_high_risk_customers():
    """Test the high risk customer identification function"""
    # Create sample RFM data
    rfm_data = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'recency': [30, 200, 300],
        'frequency': [5, 1, 1],
        'monetary': [500, 100, 50]
    })
    
    dp = DataProcessing()
    rfm_df, _, high_risk_cluster, _, _ = dp.identify_high_risk_customers(rfm_data)
    
    # The third cluster should be high risk with highest score
    assert high_risk_cluster == 2
    
    # High risk customer should have high recency, low frequency, low monetary
    assert rfm_df.loc[rfm_df['CustomerId'] == 3, 'is_high_risk'].values[0] == 1
    assert rfm_df.loc[rfm_df['CustomerId'] == 1, 'is_high_risk'].values[0] == 0


def test_merge_with_original_data():
    """Test merging high risk indicator back into original data"""
    # Sample original data
    original_data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, -50, 200]
    })
    
    # Sample RFM data with risk indicator
    rfm_data = pd.DataFrame({
        'CustomerId': [1, 2],
        'is_high_risk': [0, 1]
    })
    
    dp = DataProcessing()
    merged_df = dp.merge_with_original_data(original_data, rfm_data)
    
    # Check that all original rows are preserved
    assert len(merged_df) == len(original_data)
    
    # Check that risk indicator is correctly assigned
    assert merged_df.loc[merged_df['CustomerId'] == 1, 'is_high_risk'].nunique() == 1
    assert merged_df.loc[merged_df['CustomerId'] == 2, 'is_high_risk'].values[0] == 1


def test_prepare_training_data():
    """Test preparation of training data"""
    # Sample processed data
    processed_data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, -50, 200],
        'Value': [100, 50, 200],
        'CountryCode': [256, 256, 256],
        'ProviderId': ['P1', 'P2', 'P1'],
        'PricingStrategy': ['2', '2', '4'],
        'ProductCategory': ['airtime', 'financial_services', 'utility_bill'],
        'ChannelId': ['3', '2', '3'],
        'transaction_hour': [14, 15, 10],
        'transaction_day': [1, 2, 3],
        'transaction_month': [11, 11, 11],
        'total_transactions': [2, 2, 1],
        'avg_transaction_amount': [25, 25, 200],
        'std_transaction_amount': [75, 75, np.nan],
        'total_transaction_value': [150, 150, 200],
        'unique_product_categories': [1, 1, 1],
        'recency': [10, 10, 5],
        'frequency': [2, 2, 1],
        'monetary': [150, 150, 200],
        'is_high_risk': [0, 0, 1]
    })
    
    dp = DataProcessing()
    X, y = dp.prepare_training_data(processed_data)
    
    # Check that all selected features are present
    expected_features = [
        'Amount', 'Value', 'CountryCode', 'ProviderId',
        'PricingStrategy', 'ProductCategory', 'ChannelId',
        'transaction_hour', 'transaction_day', 'transaction_month',
        'total_transactions', 'avg_transaction_amount',
        'std_transaction_amount', 'total_transaction_value',
        'unique_product_categories', 'recency', 'frequency', 'monetary'
    ]
    
    assert all(feature in X.columns for feature in expected_features)
    assert set(y.unique()) == {0, 1}