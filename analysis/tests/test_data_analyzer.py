import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_analyzer import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):
    """Test cases for DataAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = DataAnalyzer()
        
        # Create sample datasets for testing
        self.sample_numerical_data = pd.DataFrame({
            'sales': [100, 150, 200, 175, 300, 250, 180, 220, 190, 160],
            'profit': [20, 30, 40, 35, 60, 50, 36, 44, 38, 32],
            'quantity': [10, 15, 20, 17, 30, 25, 18, 22, 19, 16]
        })
        
        self.sample_mixed_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'sales': [100, 150, 200, 175, 300, 250, 180, 220, 190, 160],
            'region': ['North', 'South', 'North', 'East', 'West', 'North', 'South', 'East', 'West', 'North']
        })
        
        self.sample_data_with_nulls = pd.DataFrame({
            'value': [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            'category': ['A', 'B', None, 'A', 'B', 'C', None, 'A', 'B', 'C']
        })
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis functionality"""
        result = self.analyzer.comprehensive_analysis(self.sample_mixed_data)
        
        # Check that all expected sections are present
        expected_sections = [
            'basic_statistics', 'data_quality', 'column_analysis',
            'correlation_analysis', 'distribution_analysis', 'time_series_analysis',
            'categorical_analysis', 'outlier_analysis', 'business_metrics', 'data_patterns'
        ]
        
        for section in expected_sections:
            self.assertIn(section, result, f"Missing section: {section}")
        
        # Check basic statistics
        self.assertIn('dataset_info', result['basic_statistics'])
        self.assertEqual(result['basic_statistics']['dataset_info']['total_rows'], 10)
        self.assertEqual(result['basic_statistics']['dataset_info']['total_columns'], 4)
    
    def test_basic_statistics(self):
        """Test basic statistics calculation"""
        result = self.analyzer._get_basic_statistics(self.sample_numerical_data)
        
        self.assertIn('dataset_info', result)
        self.assertIn('numerical_summary', result)
        self.assertIn('missing_data', result)
        self.assertIn('data_types', result)
        
        # Check dataset info
        dataset_info = result['dataset_info']
        self.assertEqual(dataset_info['total_rows'], 10)
        self.assertEqual(dataset_info['total_columns'], 3)
        self.assertEqual(dataset_info['numerical_columns'], 3)
        
        # Check numerical summary
        self.assertIn('sales', result['numerical_summary'])
        self.assertIn('mean', result['numerical_summary']['sales'])
        self.assertIn('std', result['numerical_summary']['sales'])
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        result = self.analyzer._assess_data_quality(self.sample_data_with_nulls)
        
        self.assertIn('completeness', result)
        self.assertIn('duplicates', result)
        self.assertIn('consistency', result)
        
        # Check completeness metrics
        completeness = result['completeness']
        self.assertEqual(completeness['total_missing_values'], 3)  # 2 in value + 1 in category
        self.assertGreater(completeness['missing_percentage'], 0)
        self.assertIn('value', completeness['columns_with_missing'])
        self.assertIn('category', completeness['columns_with_missing'])
    
    def test_outlier_detection(self):
        """Test outlier detection using IQR method"""
        # Create data with obvious outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is an outlier
        
        outliers = self.analyzer._detect_outliers_iqr(data_with_outliers)
        
        # Should detect the outlier value 100
        self.assertGreater(len(outliers), 0)
        self.assertIn(100, outliers.values)
    
    def test_outlier_detection_edge_cases(self):
        """Test outlier detection with edge cases"""
        # Test with empty series
        empty_series = pd.Series([])
        outliers_empty = self.analyzer._detect_outliers_iqr(empty_series)
        self.assertEqual(len(outliers_empty), 0)
        
        # Test with series containing NaN
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5])
        outliers_nan = self.analyzer._detect_outliers_iqr(series_with_nan)
        self.assertIsInstance(outliers_nan, pd.Series)
        
        # Test with all same values
        same_values = pd.Series([5, 5, 5, 5, 5])
        outliers_same = self.analyzer._detect_outliers_iqr(same_values)
        self.assertEqual(len(outliers_same), 0)
    
    def test_numerical_column_analysis(self):
        """Test numerical column analysis"""
        sales_data = self.sample_numerical_data['sales']
        result = self.analyzer._analyze_numerical_column(sales_data)
        
        expected_keys = [
            'min', 'max', 'mean', 'median', 'std', 'range',
            'outliers', 'distribution_type', 'is_normal',
            'zero_values', 'negative_values', 'positive_values'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Check specific values
        self.assertEqual(result['min'], 100)
        self.assertEqual(result['max'], 300)
        self.assertEqual(result['positive_values'], 10)
        self.assertEqual(result['negative_values'], 0)
        self.assertEqual(result['zero_values'], 0)
    
    def test_text_column_analysis(self):
        """Test text/categorical column analysis"""
        category_data = self.sample_mixed_data['category']
        result = self.analyzer._analyze_text_column(category_data)
        
        expected_keys = [
            'most_frequent', 'most_frequent_count', 'least_frequent',
            'least_frequent_count', 'cardinality', 'is_high_cardinality',
            'is_binary', 'average_length', 'max_length', 'min_length',
            'contains_numbers', 'contains_special_chars'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Check specific values
        self.assertEqual(result['cardinality'], 3)  # A, B, C
        self.assertFalse(result['is_binary'])
        self.assertEqual(result['max_length'], 1)
        self.assertEqual(result['min_length'], 1)
    
    def test_datetime_column_analysis(self):
        """Test datetime column analysis"""
        date_data = self.sample_mixed_data['date']
        result = self.analyzer._analyze_datetime_column(date_data)
        
        expected_keys = [
            'earliest_date', 'latest_date', 'date_range_days',
            'unique_dates', 'most_common_date', 'year_range',
            'month_distribution', 'day_of_week_distribution', 'has_time_component'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Check specific values
        self.assertEqual(result['date_range_days'], 9)  # 10 days - 1
        self.assertEqual(result['unique_dates'], 10)
        self.assertFalse(result['has_time_component'])  # No time component in date range
    
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        result = self.analyzer._analyze_correlations(self.sample_numerical_data)
        
        self.assertIn('correlation_matrix', result)
        self.assertIn('strong_correlations', result)
        self.assertIn('correlation_insights', result)
        
        # Check correlation matrix structure
        corr_matrix = result['correlation_matrix']
        self.assertIn('sales', corr_matrix)
        self.assertIn('profit', corr_matrix)
        self.assertIn('quantity', corr_matrix)
    
    def test_normality_testing(self):
        """Test normality testing"""
        # Test with normal-ish data
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        is_normal = self.analyzer._test_normality(normal_data)
        self.assertIsInstance(is_normal, bool)
        
        # Test with clearly non-normal data
        non_normal_data = pd.Series([1, 1, 1, 1, 1, 100, 100, 100, 100, 100])
        is_normal_non = self.analyzer._test_normality(non_normal_data)
        self.assertIsInstance(is_normal_non, bool)
    
    def test_empty_dataframe(self):
        """Test behavior with empty dataframe"""
        empty_df = pd.DataFrame()
        result = self.analyzer.comprehensive_analysis(empty_df)
        
        # Should not crash and should return some structure
        self.assertIsInstance(result, dict)
        self.assertIn('basic_statistics', result)
    
    def test_single_column_dataframe(self):
        """Test behavior with single column dataframe"""
        single_col_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = self.analyzer.comprehensive_analysis(single_col_df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('basic_statistics', result)
        self.assertEqual(result['basic_statistics']['dataset_info']['total_columns'], 1)
    
    def test_convert_numpy_types(self):
        """Test numpy type conversion"""
        # Test with various numpy types
        test_data = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3]),
            'nested': {
                'inner_int': np.int32(10),
                'inner_float': np.float32(2.5)
            }
        }
        
        result = self.analyzer._convert_numpy_types(test_data)
        
        # Check that numpy types are converted to Python types
        self.assertIsInstance(result['int64'], int)
        self.assertIsInstance(result['float64'], float)
        self.assertIsInstance(result['bool'], bool)
        self.assertIsInstance(result['array'], list)
        self.assertIsInstance(result['nested']['inner_int'], int)
        self.assertIsInstance(result['nested']['inner_float'], float)


if __name__ == '__main__':
    unittest.main() 