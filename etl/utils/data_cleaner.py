import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Dynamic data cleaning utility with various cleaning strategies"""
    
    def __init__(self):
        self.cleaning_log = []
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_values(self, df, columns=None, strategy='drop', fill_value=None):
        """Handle missing values with various strategies"""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.columns
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
            elif strategy == 'fill_mean':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].mean())
            elif strategy == 'fill_median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].median())
            elif strategy == 'fill_mode':
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
            elif strategy == 'forward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            elif strategy == 'backward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_cleaned[col] = df_cleaned[col].interpolate()
            elif strategy == 'fill_value':
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            elif strategy == 'fill_zero':
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        return df_cleaned
    
    def remove_duplicates(self, df, subset=None, keep='first'):
        """Remove duplicate rows"""
        df_cleaned = df.copy()
        
        if subset is None:
            df_cleaned = df_cleaned.drop_duplicates(keep=keep)
        else:
            df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep=keep)
        
        return df_cleaned
    
    def handle_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """Handle outliers using various methods"""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Remove outliers
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & 
                                      (df_cleaned[col] <= upper_bound)]
            
            elif method == 'z_score':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df_cleaned = df_cleaned[z_scores <= 3]
            
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df[[col]].dropna())
                df_cleaned = df_cleaned[outliers == 1]
            
            elif method == 'percentile':
                lower_percentile = df[col].quantile(0.01)
                upper_percentile = df[col].quantile(0.99)
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_percentile) & 
                                      (df_cleaned[col] <= upper_percentile)]
            
            elif method == 'cap':
                # Cap outliers instead of removing them
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_cleaned
    
    def standardize_text(self, df, columns=None, operations=None):
        """Standardize text data"""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        if operations is None:
            operations = ['lowercase', 'strip', 'normalize_whitespace']
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if 'lowercase' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower()
            
            if 'uppercase' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.upper()
            
            if 'strip' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
            if 'remove_special_chars' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
            
            if 'normalize_whitespace' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\s+', ' ', regex=True)
            
            if 'remove_digits' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\d+', '', regex=True)
            
            if 'title_case' in operations:
                df_cleaned[col] = df_cleaned[col].astype(str).str.title()
        
        return df_cleaned
    
    def convert_data_types(self, df, type_mapping):
        """Convert data types based on mapping"""
        df_cleaned = df.copy()
        
        for col, target_type in type_mapping.items():
            if col not in df.columns:
                continue
            
            try:
                if target_type == 'datetime':
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                elif target_type == 'numeric':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                elif target_type == 'categorical':
                    df_cleaned[col] = df_cleaned[col].astype('category')
                elif target_type == 'string':
                    df_cleaned[col] = df_cleaned[col].astype(str)
                elif target_type == 'int':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                elif target_type == 'float':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(float)
                elif target_type == 'bool':
                    df_cleaned[col] = df_cleaned[col].astype(bool)
            except Exception as e:
                print(f"Error converting {col} to {target_type}: {str(e)}")
        
        return df_cleaned
    
    def handle_categorical_data(self, df, columns=None, encoding='label', drop_original=True):
        """Handle categorical data with various encoding methods"""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if encoding == 'label':
                le = LabelEncoder()
                df_cleaned[f'{col}_encoded'] = le.fit_transform(df_cleaned[col].astype(str))
                self.encoders[f'{col}_label'] = le
                
                if drop_original:
                    df_cleaned = df_cleaned.drop(columns=[col])
            
            elif encoding == 'onehot':
                # Get dummies
                dummies = pd.get_dummies(df_cleaned[col], prefix=col)
                df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
                
                if drop_original:
                    df_cleaned = df_cleaned.drop(columns=[col])
            
            elif encoding == 'frequency':
                # Frequency encoding
                freq_map = df_cleaned[col].value_counts().to_dict()
                df_cleaned[f'{col}_freq'] = df_cleaned[col].map(freq_map)
                
                if drop_original:
                    df_cleaned = df_cleaned.drop(columns=[col])
            
            elif encoding == 'target':
                # This would need target variable - placeholder for now
                pass
        
        return df_cleaned
    
    def normalize_numerical_data(self, df, columns=None, method='standard'):
        """Normalize numerical data"""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
                df_cleaned[col] = scaler.fit_transform(df_cleaned[[col]])
                self.scalers[f'{col}_standard'] = scaler
            
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df_cleaned[col] = scaler.fit_transform(df_cleaned[[col]])
                self.scalers[f'{col}_minmax'] = scaler
            
            elif method == 'robust':
                scaler = RobustScaler()
                df_cleaned[col] = scaler.fit_transform(df_cleaned[[col]])
                self.scalers[f'{col}_robust'] = scaler
            
            elif method == 'log':
                # Log transformation (handle negative values)
                df_cleaned[col] = np.log1p(df_cleaned[col] - df_cleaned[col].min() + 1)
            
            elif method == 'sqrt':
                # Square root transformation
                df_cleaned[col] = np.sqrt(df_cleaned[col] - df_cleaned[col].min() + 1)
        
        return df_cleaned
    
    def validate_data_consistency(self, df, rules=None):
        """Validate data consistency based on rules"""
        df_cleaned = df.copy()
        validation_issues = []
        
        if rules is None:
            rules = []
        
        for rule in rules:
            rule_type = rule.get('type')
            
            if rule_type == 'range_check':
                col = rule['column']
                min_val = rule.get('min')
                max_val = rule.get('max')
                
                if col in df.columns:
                    if min_val is not None:
                        invalid_rows = df_cleaned[df_cleaned[col] < min_val]
                        if len(invalid_rows) > 0:
                            validation_issues.append(f"{len(invalid_rows)} rows in {col} below minimum {min_val}")
                    
                    if max_val is not None:
                        invalid_rows = df_cleaned[df_cleaned[col] > max_val]
                        if len(invalid_rows) > 0:
                            validation_issues.append(f"{len(invalid_rows)} rows in {col} above maximum {max_val}")
            
            elif rule_type == 'format_check':
                col = rule['column']
                pattern = rule['pattern']
                
                if col in df.columns:
                    invalid_rows = df_cleaned[~df_cleaned[col].astype(str).str.match(pattern)]
                    if len(invalid_rows) > 0:
                        validation_issues.append(f"{len(invalid_rows)} rows in {col} don't match pattern {pattern}")
        
        return df_cleaned, validation_issues
    
    def remove_constant_columns(self, df, threshold=0.95):
        """Remove columns with constant or near-constant values"""
        df_cleaned = df.copy()
        columns_to_drop = []
        
        for col in df.columns:
            # Check if column has very low variance
            if df[col].nunique() == 1:
                columns_to_drop.append(col)
            elif df[col].dtype in ['object', 'category']:
                # For categorical, check if one value dominates
                value_counts = df[col].value_counts()
                if len(value_counts) > 0 and value_counts.iloc[0] / len(df) > threshold:
                    columns_to_drop.append(col)
        
        if columns_to_drop:
            df_cleaned = df_cleaned.drop(columns=columns_to_drop)
        
        return df_cleaned
    
    def handle_date_columns(self, df, date_columns=None, extract_features=True):
        """Handle date columns and extract features"""
        df_cleaned = df.copy()
        
        if date_columns is None:
            # Try to detect date columns
            date_columns = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    date_columns.append(col)
                elif df[col].dtype == 'object':
                    # Try to parse as date
                    try:
                        pd.to_datetime(df[col].head(10))
                        date_columns.append(col)
                    except:
                        pass
        
        for col in date_columns:
            if col not in df.columns:
                continue
            
            # Convert to datetime
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            
            if extract_features:
                # Extract date features
                df_cleaned[f'{col}_year'] = df_cleaned[col].dt.year
                df_cleaned[f'{col}_month'] = df_cleaned[col].dt.month
                df_cleaned[f'{col}_day'] = df_cleaned[col].dt.day
                df_cleaned[f'{col}_dayofweek'] = df_cleaned[col].dt.dayofweek
                df_cleaned[f'{col}_quarter'] = df_cleaned[col].dt.quarter
                df_cleaned[f'{col}_is_weekend'] = df_cleaned[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df_cleaned
    
    def clean_column_names(self, df):
        """Clean column names"""
        df_cleaned = df.copy()
        
        # Clean column names
        new_columns = []
        for col in df.columns:
            # Convert to lowercase
            new_col = str(col).lower()
            # Replace spaces and special characters with underscores
            new_col = re.sub(r'[^\w]', '_', new_col)
            # Remove multiple underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            new_columns.append(new_col)
        
        df_cleaned.columns = new_columns
        return df_cleaned
    
    def get_cleaning_summary(self, original_df, cleaned_df):
        """Get summary of cleaning operations"""
        summary = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum(),
            'duplicates_before': original_df.duplicated().sum(),
            'duplicates_after': cleaned_df.duplicated().sum()
        }
        
        return summary 