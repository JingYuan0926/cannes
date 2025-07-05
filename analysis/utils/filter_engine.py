"""
Filter Engine Module

Provides comprehensive filtering capabilities for datasets,
with intelligent filter detection and business logic.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)

class FilterEngine:
    """
    Comprehensive filter engine for dataset filtering and data manipulation
    """
    
    def __init__(self):
        self.filter_functions = {
            'equals': self._filter_equals,
            'not_equals': self._filter_not_equals,
            'contains': self._filter_contains,
            'not_contains': self._filter_not_contains,
            'starts_with': self._filter_starts_with,
            'ends_with': self._filter_ends_with,
            'greater_than': self._filter_greater_than,
            'less_than': self._filter_less_than,
            'greater_equal': self._filter_greater_equal,
            'less_equal': self._filter_less_equal,
            'between': self._filter_between,
            'in_list': self._filter_in_list,
            'not_in_list': self._filter_not_in_list,
            'is_null': self._filter_is_null,
            'is_not_null': self._filter_is_not_null,
            'regex': self._filter_regex,
            'date_range': self._filter_date_range,
            'top_n': self._filter_top_n,
            'bottom_n': self._filter_bottom_n,
            'outliers': self._filter_outliers,
            'duplicates': self._filter_duplicates,
            'unique': self._filter_unique,
            'sample': self._filter_sample
        }
    
    def apply_filters(self, df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply multiple filters to a DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            filters (List[Dict]): List of filter configurations
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        try:
            filtered_df = df.copy()
            
            for filter_config in filters:
                filtered_df = self.apply_single_filter(filtered_df, filter_config)
                
                # Log filter application
                logger.info(f"Applied filter: {filter_config.get('type', 'unknown')} - "
                           f"Rows: {len(df)} -> {len(filtered_df)}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return df
    
    def apply_single_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a single filter to a DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            filter_config (Dict): Filter configuration
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        try:
            filter_type = filter_config.get('type')
            
            if filter_type not in self.filter_functions:
                logger.warning(f"Unknown filter type: {filter_type}")
                return df
            
            filter_func = self.filter_functions[filter_type]
            return filter_func(df, filter_config)
            
        except Exception as e:
            logger.error(f"Error applying single filter: {str(e)}")
            return df
    
    def get_available_filters(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get available filters for each column based on data types
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Available filters for each column
        """
        try:
            available_filters = {}
            
            for column in df.columns:
                column_type = df[column].dtype
                filters = []
                
                # Common filters for all types
                filters.extend(['equals', 'not_equals', 'is_null', 'is_not_null'])
                
                # String-specific filters
                if pd.api.types.is_string_dtype(column_type) or pd.api.types.is_object_dtype(column_type):
                    filters.extend(['contains', 'not_contains', 'starts_with', 'ends_with', 
                                   'regex', 'in_list', 'not_in_list'])
                
                # Numeric-specific filters
                if pd.api.types.is_numeric_dtype(column_type):
                    filters.extend(['greater_than', 'less_than', 'greater_equal', 'less_equal',
                                   'between', 'top_n', 'bottom_n', 'outliers'])
                
                # Date-specific filters
                if pd.api.types.is_datetime64_any_dtype(column_type):
                    filters.extend(['date_range', 'greater_than', 'less_than'])
                
                # Categorical filters
                if df[column].nunique() < len(df) * 0.5:  # Heuristic for categorical
                    filters.extend(['in_list', 'not_in_list'])
                
                available_filters[column] = list(set(filters))
            
            # Add dataset-level filters
            available_filters['_dataset'] = ['duplicates', 'unique', 'sample']
            
            return available_filters
            
        except Exception as e:
            logger.error(f"Error getting available filters: {str(e)}")
            return {}
    
    def intelligent_filter_suggestions(self, df: pd.DataFrame, user_query: str) -> List[Dict[str, Any]]:
        """
        Suggest filters based on user query using natural language processing
        
        Args:
            df (pd.DataFrame): Input DataFrame
            user_query (str): User's natural language query
            
        Returns:
            List[Dict]: Suggested filter configurations
        """
        try:
            suggestions = []
            query_lower = user_query.lower()
            
            # Extract column names mentioned in query
            mentioned_columns = []
            for col in df.columns:
                if col.lower() in query_lower or col.lower().replace('_', ' ') in query_lower:
                    mentioned_columns.append(col)
            
            # Pattern matching for different filter types
            patterns = {
                'equals': [r'equals?', r'is\s+exactly', r'exactly\s+equals?'],
                'contains': [r'contains?', r'includes?', r'has\s+.*'],
                'greater_than': [r'greater\s+than', r'more\s+than', r'above', r'>\s*'],
                'less_than': [r'less\s+than', r'below', r'under', r'<\s*'],
                'between': [r'between\s+.*\s+and', r'from\s+.*\s+to'],
                'date_range': [r'from\s+\d{4}', r'between\s+\d{4}', r'in\s+\d{4}'],
                'top_n': [r'top\s+\d+', r'highest\s+\d+', r'best\s+\d+'],
                'bottom_n': [r'bottom\s+\d+', r'lowest\s+\d+', r'worst\s+\d+'],
                'outliers': [r'outliers?', r'anomalies', r'unusual'],
                'duplicates': [r'duplicates?', r'repeated', r'duplicate\s+records'],
                'sample': [r'sample', r'random', r'subset']
            }
            
            # Find matching patterns
            for filter_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, query_lower):
                        # Extract values from query
                        values = self._extract_values_from_query(query_lower, filter_type)
                        
                        if mentioned_columns:
                            for col in mentioned_columns:
                                suggestions.append({
                                    'type': filter_type,
                                    'column': col,
                                    'values': values,
                                    'confidence': 0.8
                                })
                        else:
                            suggestions.append({
                                'type': filter_type,
                                'values': values,
                                'confidence': 0.6
                            })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating filter suggestions: {str(e)}")
            return []
    
    def _extract_values_from_query(self, query: str, filter_type: str) -> List[Any]:
        """Extract values from natural language query"""
        try:
            values = []
            
            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if numbers:
                values.extend([float(n) if '.' in n else int(n) for n in numbers])
            
            # Extract quoted strings
            quoted_strings = re.findall(r'"([^"]*)"', query)
            if quoted_strings:
                values.extend(quoted_strings)
            
            # Extract dates
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}/\d{1,2}/\d{4}'
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, query)
                if dates:
                    values.extend(dates)
            
            return values
            
        except Exception as e:
            logger.error(f"Error extracting values: {str(e)}")
            return []
    
    # Filter implementation functions
    def _filter_equals(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column equals value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] == value]
    
    def _filter_not_equals(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column does not equal value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] != value]
    
    def _filter_contains(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column contains value"""
        column = config.get('column')
        value = config.get('value')
        case_sensitive = config.get('case_sensitive', False)
        
        if column not in df.columns:
            return df
        
        if case_sensitive:
            return df[df[column].astype(str).str.contains(str(value), na=False)]
        else:
            return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
    
    def _filter_not_contains(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column does not contain value"""
        column = config.get('column')
        value = config.get('value')
        case_sensitive = config.get('case_sensitive', False)
        
        if column not in df.columns:
            return df
        
        if case_sensitive:
            return df[~df[column].astype(str).str.contains(str(value), na=False)]
        else:
            return df[~df[column].astype(str).str.contains(str(value), case=False, na=False)]
    
    def _filter_starts_with(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column starts with value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column].astype(str).str.startswith(str(value), na=False)]
    
    def _filter_ends_with(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column ends with value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column].astype(str).str.endswith(str(value), na=False)]
    
    def _filter_greater_than(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is greater than value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] > value]
    
    def _filter_less_than(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is less than value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] < value]
    
    def _filter_greater_equal(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is greater than or equal to value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] >= value]
    
    def _filter_less_equal(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is less than or equal to value"""
        column = config.get('column')
        value = config.get('value')
        
        if column not in df.columns:
            return df
        
        return df[df[column] <= value]
    
    def _filter_between(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is between two values"""
        column = config.get('column')
        min_value = config.get('min_value')
        max_value = config.get('max_value')
        
        if column not in df.columns:
            return df
        
        return df[(df[column] >= min_value) & (df[column] <= max_value)]
    
    def _filter_in_list(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column value is in list"""
        column = config.get('column')
        values = config.get('values', [])
        
        if column not in df.columns:
            return df
        
        return df[df[column].isin(values)]
    
    def _filter_not_in_list(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column value is not in list"""
        column = config.get('column')
        values = config.get('values', [])
        
        if column not in df.columns:
            return df
        
        return df[~df[column].isin(values)]
    
    def _filter_is_null(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is null"""
        column = config.get('column')
        
        if column not in df.columns:
            return df
        
        return df[df[column].isnull()]
    
    def _filter_is_not_null(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column is not null"""
        column = config.get('column')
        
        if column not in df.columns:
            return df
        
        return df[df[column].notnull()]
    
    def _filter_regex(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where column matches regex pattern"""
        column = config.get('column')
        pattern = config.get('pattern')
        
        if column not in df.columns:
            return df
        
        return df[df[column].astype(str).str.match(pattern, na=False)]
    
    def _filter_date_range(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter rows where date column is within range"""
        column = config.get('column')
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if column not in df.columns:
            return df
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        return df[(df[column] >= start_date) & (df[column] <= end_date)]
    
    def _filter_top_n(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter top N rows based on column value"""
        column = config.get('column')
        n = config.get('n', 10)
        
        if column not in df.columns:
            return df
        
        return df.nlargest(n, column)
    
    def _filter_bottom_n(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter bottom N rows based on column value"""
        column = config.get('column')
        n = config.get('n', 10)
        
        if column not in df.columns:
            return df
        
        return df.nsmallest(n, column)
    
    def _filter_outliers(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter outliers based on statistical methods"""
        column = config.get('column')
        method = config.get('method', 'iqr')  # 'iqr' or 'zscore'
        threshold = config.get('threshold', 1.5)
        
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            return df[z_scores > threshold]
        
        return df
    
    def _filter_duplicates(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter duplicate rows"""
        columns = config.get('columns')  # Specific columns to check for duplicates
        keep = config.get('keep', 'first')  # 'first', 'last', or False
        
        if columns:
            return df[df.duplicated(subset=columns, keep=keep)]
        else:
            return df[df.duplicated(keep=keep)]
    
    def _filter_unique(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter unique rows (remove duplicates)"""
        columns = config.get('columns')  # Specific columns to check for duplicates
        keep = config.get('keep', 'first')  # 'first', 'last', or False
        
        if columns:
            return df.drop_duplicates(subset=columns, keep=keep)
        else:
            return df.drop_duplicates(keep=keep)
    
    def _filter_sample(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter random sample of rows"""
        n = config.get('n')  # Number of rows
        frac = config.get('frac')  # Fraction of rows
        random_state = config.get('random_state', 42)
        
        if n:
            return df.sample(n=min(n, len(df)), random_state=random_state)
        elif frac:
            return df.sample(frac=frac, random_state=random_state)
        else:
            return df.sample(n=min(100, len(df)), random_state=random_state)
    
    def get_filter_summary(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of filter application
        
        Args:
            original_df (pd.DataFrame): Original DataFrame
            filtered_df (pd.DataFrame): Filtered DataFrame
            
        Returns:
            Dict: Filter summary statistics
        """
        try:
            return {
                'original_rows': len(original_df),
                'filtered_rows': len(filtered_df),
                'rows_removed': len(original_df) - len(filtered_df),
                'percentage_kept': (len(filtered_df) / len(original_df)) * 100 if len(original_df) > 0 else 0,
                'columns': len(filtered_df.columns),
                'filter_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating filter summary: {str(e)}")
            return {} 