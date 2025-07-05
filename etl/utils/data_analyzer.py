import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis for quality assessment and cleaning recommendations"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def comprehensive_analysis(self, df):
        """Perform comprehensive analysis of the dataset"""
        analysis = {
            'basic_info': self._get_basic_info(df),
            'missing_data': self._analyze_missing_data(df),
            'duplicates': self._analyze_duplicates(df),
            'outliers': self._detect_outliers(df),
            'data_quality': self._assess_data_quality(df),
            'statistical_summary': self._get_statistical_summary(df),
            'data_types': self._analyze_data_types(df),
            'categorical_analysis': self._analyze_categorical_data(df),
            'correlations': self._analyze_correlations(df)
        }
        
        # Convert all numpy types to Python types
        return self._convert_numpy_types(analysis)
    
    def _get_basic_info(self, df):
        """Get basic information about the dataset"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'size': df.size
        }
    
    def _analyze_missing_data(self, df):
        """Analyze missing data patterns"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2),
                    'data_type': str(df[col].dtype),
                    'non_null_count': int(df[col].count())
                }
        
        return missing_info
    
    def _analyze_duplicates(self, df):
        """Analyze duplicate rows"""
        total_duplicates = df.duplicated().sum()
        duplicate_percentage = (total_duplicates / len(df)) * 100
        
        # Check for duplicates in specific columns
        column_duplicates = {}
        for col in df.columns:
            col_duplicates = df[col].duplicated().sum()
            if col_duplicates > 0:
                column_duplicates[col] = {
                    'duplicate_count': int(col_duplicates),
                    'duplicate_percentage': round((col_duplicates / len(df)) * 100, 2)
                }
        
        return {
            'total_duplicates': int(total_duplicates),
            'duplicate_percentage': round(duplicate_percentage, 2),
            'column_duplicates': column_duplicates
        }
    
    def _detect_outliers(self, df):
        """Detect outliers in numerical columns"""
        outliers_info = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].count() > 0:  # Only analyze if column has data
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                z_outliers = (z_scores > 3).sum()
                
                outliers_info[col] = {
                    'iqr_outliers': int(iqr_outliers),
                    'iqr_percentage': round((iqr_outliers / len(df)) * 100, 2),
                    'z_score_outliers': int(z_outliers),
                    'z_score_percentage': round((z_outliers / len(df)) * 100, 2),
                    'min_value': float(df[col].min()),
                    'max_value': float(df[col].max()),
                    'mean_value': float(df[col].mean()),
                    'std_value': float(df[col].std())
                }
        
        return outliers_info
    
    def _assess_data_quality(self, df):
        """Assess overall data quality issues"""
        quality_issues = {}
        
        for col in df.columns:
            issues = []
            
            # Check for inconsistent data types
            if df[col].dtype == 'object':
                # Check for mixed types
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types = set(type(val).__name__ for val in sample_values)
                    if len(types) > 1:
                        issues.append(f"Mixed data types: {types}")
                
                # Check for inconsistent formatting
                if any(df[col].astype(str).str.contains(r'^\s+|\s+$', na=False)):
                    issues.append("Contains leading/trailing whitespace")
                
                # Check for special characters
                if any(df[col].astype(str).str.contains(r'[^\w\s]', na=False)):
                    issues.append("Contains special characters")
            
            # Check for constant values
            if df[col].nunique() == 1:
                issues.append("Constant values (no variation)")
            
            # Check for high cardinality in categorical columns
            if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.9:
                issues.append("Very high cardinality (possibly unique identifiers)")
            
            if issues:
                quality_issues[col] = issues
        
        return quality_issues
    
    def _get_statistical_summary(self, df):
        """Get statistical summary of the dataset"""
        try:
            # Get basic statistics
            desc = df.describe(include='all').to_dict()
            
            # Convert numpy types to Python types for JSON serialization
            for col in desc:
                for stat in desc[col]:
                    if pd.isna(desc[col][stat]):
                        desc[col][stat] = None
                    elif isinstance(desc[col][stat], (np.integer, np.floating)):
                        desc[col][stat] = float(desc[col][stat])
            
            return desc
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_data_types(self, df):
        """Analyze data types and suggest improvements"""
        type_analysis = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            suggestions = []
            
            if current_type == 'object':
                # Check if it could be datetime
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    suggestions.append('datetime')
                except:
                    pass
                
                # Check if it could be numeric
                try:
                    pd.to_numeric(df[col].dropna().head(10))
                    suggestions.append('numeric')
                except:
                    pass
                
                # Check if it could be categorical
                if df[col].nunique() < len(df) * 0.5:
                    suggestions.append('categorical')
            
            type_analysis[col] = {
                'current_type': current_type,
                'suggested_types': suggestions,
                'unique_values': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            }
        
        return type_analysis
    
    def _analyze_categorical_data(self, df):
        """Analyze categorical columns"""
        categorical_analysis = {}
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'cardinality_ratio': round(df[col].nunique() / len(df), 4)
            }
        
        return categorical_analysis
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numerical columns"""
        try:
            numerical_df = df.select_dtypes(include=[np.number])
            
            if len(numerical_df.columns) < 2:
                return {"message": "Not enough numerical columns for correlation analysis"}
            
            corr_matrix = numerical_df.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            'column1': corr_matrix.columns[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': round(float(corr_val), 4)
                        })
            
            return {
                'high_correlations': high_correlations,
                'correlation_matrix_shape': corr_matrix.shape
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_cleaning_priority(self, analysis):
        """Determine cleaning priority based on analysis results"""
        priorities = []
        
        # High priority: Missing data > 50%
        for col, info in analysis.get('missing_data', {}).items():
            if info['missing_percentage'] > 50:
                priorities.append({
                    'issue': f'High missing data in {col}',
                    'priority': 10,
                    'action': 'consider_dropping_column'
                })
        
        # High priority: Duplicates > 10%
        if analysis.get('duplicates', {}).get('duplicate_percentage', 0) > 10:
            priorities.append({
                'issue': 'High duplicate percentage',
                'priority': 9,
                'action': 'remove_duplicates'
            })
        
        # Medium priority: Outliers > 5%
        for col, info in analysis.get('outliers', {}).items():
            if info['iqr_percentage'] > 5:
                priorities.append({
                    'issue': f'High outliers in {col}',
                    'priority': 6,
                    'action': 'handle_outliers'
                })
        
        # Medium priority: Data quality issues
        for col, issues in analysis.get('data_quality', {}).items():
            for issue in issues:
                priorities.append({
                    'issue': f'{issue} in {col}',
                    'priority': 5,
                    'action': 'standardize_data'
                })
        
        return sorted(priorities, key=lambda x: x['priority'], reverse=True) 