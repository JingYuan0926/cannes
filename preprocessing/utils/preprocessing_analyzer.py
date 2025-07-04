import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PreprocessingAnalyzer:
    """Comprehensive preprocessing analysis for datasets"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_) or isinstance(obj, np.unicode_):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # Handle any remaining numpy scalars
            return obj.item()
        else:
            return obj
    
    def comprehensive_analysis(self, df):
        """Perform comprehensive preprocessing analysis of the dataset"""
        analysis = {
            'basic_info': self._get_basic_info(df),
            'numerical_analysis': self._analyze_numerical_columns(df),
            'categorical_analysis': self._analyze_categorical_columns(df),
            'distribution_analysis': self._analyze_distributions(df),
            'scaling_requirements': self._analyze_scaling_requirements(df),
            'encoding_requirements': self._analyze_encoding_requirements(df),
            'feature_engineering': self._analyze_feature_engineering_opportunities(df),
            'data_quality': self._analyze_data_quality(df),
            'correlation_analysis': self._analyze_correlations(df),
            'preprocessing_recommendations': self._get_preprocessing_recommendations(df)
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
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def _analyze_numerical_columns(self, df):
        """Analyze numerical columns for preprocessing needs"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_analysis = {}
        
        for col in numerical_cols:
            if df[col].count() > 0:  # Only analyze if column has data
                col_data = df[col].dropna()
                
                # Basic statistics
                stats_info = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'variance': col_data.var(),
                    'coefficient_of_variation': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                }
                
                # Distribution analysis
                distribution_info = {
                    'is_normal': self._test_normality(col_data),
                    'has_outliers': self._detect_outliers_iqr(col_data),
                    'outlier_count': self._count_outliers_iqr(col_data),
                    'zero_count': (col_data == 0).sum(),
                    'negative_count': (col_data < 0).sum(),
                    'unique_values': col_data.nunique(),
                    'unique_ratio': col_data.nunique() / len(col_data)
                }
                
                # Scaling requirements
                scaling_info = {
                    'needs_scaling': self._needs_scaling(col_data),
                    'recommended_scaler': self._recommend_scaler(col_data),
                    'scale_factor': col_data.max() / col_data.min() if col_data.min() != 0 else float('inf')
                }
                
                numerical_analysis[col] = {
                    'statistics': stats_info,
                    'distribution': distribution_info,
                    'scaling': scaling_info
                }
        
        return numerical_analysis
    
    def _analyze_categorical_columns(self, df):
        """Analyze categorical columns for preprocessing needs"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_analysis = {}
        
        for col in categorical_cols:
            if df[col].count() > 0:  # Only analyze if column has data
                col_data = df[col].dropna()
                
                # Basic statistics
                stats_info = {
                    'unique_count': col_data.nunique(),
                    'unique_ratio': col_data.nunique() / len(col_data),
                    'most_frequent': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                    'most_frequent_count': col_data.value_counts().iloc[0] if len(col_data) > 0 else 0,
                    'least_frequent': col_data.value_counts().index[-1] if len(col_data.value_counts()) > 0 else None,
                    'least_frequent_count': col_data.value_counts().iloc[-1] if len(col_data.value_counts()) > 0 else 0
                }
                
                # Cardinality analysis
                cardinality_info = {
                    'cardinality_type': self._classify_cardinality(col_data),
                    'is_high_cardinality': col_data.nunique() > len(col_data) * 0.5,
                    'is_binary': col_data.nunique() == 2,
                    'is_ordinal': self._detect_ordinal(col_data),
                    'has_missing_category': col_data.isnull().sum() > 0
                }
                
                # Encoding requirements
                encoding_info = {
                    'recommended_encoding': self._recommend_encoding(col_data),
                    'can_be_label_encoded': col_data.nunique() < 50,
                    'should_be_one_hot': col_data.nunique() <= 10 and col_data.nunique() > 2,
                    'needs_target_encoding': col_data.nunique() > 10
                }
                
                categorical_analysis[col] = {
                    'statistics': stats_info,
                    'cardinality': cardinality_info,
                    'encoding': encoding_info
                }
        
        return categorical_analysis
    
    def _analyze_distributions(self, df):
        """Analyze data distributions for preprocessing needs"""
        distribution_analysis = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].count() > 0:
                col_data = df[col].dropna()
                
                distribution_analysis[col] = {
                    'distribution_type': self._identify_distribution(col_data),
                    'normality_test': self._test_normality(col_data),
                    'transformation_needed': self._needs_transformation(col_data),
                    'recommended_transformation': self._recommend_transformation(col_data),
                    'skewness_level': self._classify_skewness(col_data.skew()),
                    'has_heavy_tails': abs(col_data.kurtosis()) > 3
                }
        
        return distribution_analysis
    
    def _analyze_scaling_requirements(self, df):
        """Analyze scaling requirements for numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        scaling_requirements = {}
        
        if len(numerical_cols) > 1:
            # Calculate scale differences between features
            ranges = {}
            means = {}
            stds = {}
            
            for col in numerical_cols:
                if df[col].count() > 0:
                    col_data = df[col].dropna()
                    ranges[col] = col_data.max() - col_data.min()
                    means[col] = col_data.mean()
                    stds[col] = col_data.std()
            
            # Determine if scaling is needed
            max_range = max(ranges.values()) if ranges else 0
            min_range = min(ranges.values()) if ranges else 0
            range_ratio = max_range / min_range if min_range != 0 else float('inf')
            
            scaling_requirements = {
                'needs_scaling': range_ratio > 10,
                'range_ratio': range_ratio,
                'different_scales': range_ratio > 10,
                'recommended_method': self._recommend_scaling_method(df[numerical_cols]),
                'features_needing_scaling': [col for col in numerical_cols if ranges.get(col, 0) > 1000 or ranges.get(col, 0) < 0.01],
                'scale_statistics': {
                    'ranges': ranges,
                    'means': means,
                    'standard_deviations': stds
                }
            }
        
        return scaling_requirements
    
    def _analyze_encoding_requirements(self, df):
        """Analyze encoding requirements for categorical features"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        encoding_requirements = {}
        
        for col in categorical_cols:
            if df[col].count() > 0:
                col_data = df[col].dropna()
                unique_count = col_data.nunique()
                
                encoding_requirements[col] = {
                    'encoding_type': self._recommend_encoding(col_data),
                    'unique_categories': unique_count,
                    'cardinality_level': self._classify_cardinality(col_data),
                    'memory_impact': self._estimate_encoding_memory_impact(col_data),
                    'is_ordinal': self._detect_ordinal(col_data),
                    'frequency_distribution': col_data.value_counts().head(10).to_dict()
                }
        
        return encoding_requirements
    
    def _analyze_feature_engineering_opportunities(self, df):
        """Analyze feature engineering opportunities"""
        opportunities = {
            'datetime_features': [],
            'text_features': [],
            'interaction_opportunities': [],
            'polynomial_opportunities': [],
            'binning_opportunities': [],
            'ratio_features': [],
            'aggregation_opportunities': []
        }
        
        # Detect datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    opportunities['datetime_features'].append({
                        'column': col,
                        'extractable_features': ['year', 'month', 'day', 'weekday', 'hour', 'quarter']
                    })
                except:
                    pass
        
        # Detect text columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object' and col not in [item['column'] for item in opportunities['datetime_features']]:
                sample_text = df[col].dropna().head(5).tolist()
                avg_length = np.mean([len(str(text)) for text in sample_text])
                if avg_length > 20:  # Likely text data
                    opportunities['text_features'].append({
                        'column': col,
                        'avg_length': avg_length,
                        'processing_methods': ['tfidf', 'count_vectorizer', 'text_cleaning']
                    })
        
        # Detect interaction opportunities
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    correlation = df[col1].corr(df[col2])
                    if abs(correlation) > 0.3:
                        opportunities['interaction_opportunities'].append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': correlation,
                            'interaction_types': ['multiply', 'add', 'ratio']
                        })
        
        # Detect polynomial opportunities
        for col in numerical_cols:
            if df[col].count() > 0:
                col_data = df[col].dropna()
                if col_data.nunique() > 10 and col_data.std() > 0:
                    opportunities['polynomial_opportunities'].append({
                        'column': col,
                        'recommended_degree': 2,
                        'reason': 'Non-linear relationships possible'
                    })
        
        # Detect binning opportunities
        for col in numerical_cols:
            if df[col].count() > 0:
                col_data = df[col].dropna()
                if col_data.nunique() > 50:
                    opportunities['binning_opportunities'].append({
                        'column': col,
                        'recommended_bins': min(10, int(np.sqrt(col_data.nunique()))),
                        'binning_method': 'equal_width' if col_data.std() > 0 else 'equal_frequency'
                    })
        
        return opportunities
    
    def _analyze_data_quality(self, df):
        """Analyze data quality issues"""
        quality_issues = {
            'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [col for col in df.select_dtypes(include=['object']).columns 
                                       if df[col].nunique() > len(df) * 0.5],
            'potential_id_columns': [col for col in df.columns if df[col].nunique() == len(df)],
            'columns_with_outliers': []
        }
        
        # Check for outliers in numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].count() > 0 and self._detect_outliers_iqr(df[col].dropna()):
                quality_issues['columns_with_outliers'].append(col)
        
        return quality_issues
    
    def _analyze_correlations(self, df):
        """Analyze correlations between features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'high_correlations': high_correlations,
                'correlation_matrix_shape': correlation_matrix.shape,
                'multicollinearity_detected': len(high_correlations) > 0
            }
        
        return {'message': 'Not enough numerical columns for correlation analysis'}
    
    def _get_preprocessing_recommendations(self, df):
        """Get high-level preprocessing recommendations"""
        recommendations = {
            'priority_actions': [],
            'optional_actions': [],
            'feature_engineering_actions': []
        }
        
        # Priority actions
        if df.isnull().sum().sum() > 0:
            recommendations['priority_actions'].append('Handle missing values')
        
        if df.duplicated().sum() > 0:
            recommendations['priority_actions'].append('Remove duplicate rows')
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            ranges = [df[col].max() - df[col].min() for col in numerical_cols if df[col].count() > 0]
            if len(ranges) > 1 and max(ranges) / min(ranges) > 10:
                recommendations['priority_actions'].append('Scale numerical features')
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            recommendations['priority_actions'].append('Encode categorical features')
        
        # Optional actions
        for col in numerical_cols:
            if df[col].count() > 0 and self._detect_outliers_iqr(df[col].dropna()):
                recommendations['optional_actions'].append(f'Handle outliers in {col}')
        
        # Feature engineering actions
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    recommendations['feature_engineering_actions'].append(f'Extract datetime features from {col}')
                except:
                    pass
        
        return recommendations
    
    # Helper methods
    def _test_normality(self, data):
        """Test if data follows normal distribution"""
        if len(data) < 8:
            return False
        try:
            _, p_value = stats.shapiro(data.sample(min(5000, len(data))))
            return p_value > 0.05
        except:
            return False
    
    def _detect_outliers_iqr(self, data):
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((data < lower_bound) | (data > upper_bound)).any()
    
    def _count_outliers_iqr(self, data):
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((data < lower_bound) | (data > upper_bound)).sum()
    
    def _needs_scaling(self, data):
        """Determine if data needs scaling"""
        return data.std() > 1000 or data.std() < 0.01 or data.max() > 1000 or data.min() < -1000
    
    def _recommend_scaler(self, data):
        """Recommend appropriate scaler"""
        if self._detect_outliers_iqr(data):
            return 'robust'
        elif data.min() >= 0:
            return 'min_max'
        else:
            return 'standard'
    
    def _classify_cardinality(self, data):
        """Classify cardinality level"""
        unique_ratio = data.nunique() / len(data)
        if unique_ratio > 0.5:
            return 'high'
        elif unique_ratio > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _detect_ordinal(self, data):
        """Detect if categorical data is ordinal"""
        # Simple heuristic: check if categories look like ordered values
        categories = data.unique()
        if len(categories) <= 2:
            return False
        
        # Check for common ordinal patterns
        ordinal_patterns = ['low', 'medium', 'high', 'small', 'large', 'bad', 'good', 'excellent']
        category_str = ' '.join([str(cat).lower() for cat in categories])
        
        return any(pattern in category_str for pattern in ordinal_patterns)
    
    def _recommend_encoding(self, data):
        """Recommend encoding method"""
        unique_count = data.nunique()
        
        if unique_count == 2:
            return 'binary'
        elif unique_count <= 5:
            return 'one_hot'
        elif unique_count <= 20:
            return 'label_encoding'
        else:
            return 'target_encoding'
    
    def _identify_distribution(self, data):
        """Identify distribution type"""
        if data.min() >= 0 and data.skew() > 1:
            return 'right_skewed'
        elif data.skew() < -1:
            return 'left_skewed'
        elif abs(data.skew()) <= 0.5:
            return 'normal'
        else:
            return 'unknown'
    
    def _needs_transformation(self, data):
        """Determine if data needs transformation"""
        return abs(data.skew()) > 1 or data.kurtosis() > 3
    
    def _recommend_transformation(self, data):
        """Recommend transformation method"""
        if data.min() > 0 and data.skew() > 1:
            return 'log'
        elif data.skew() > 1:
            return 'sqrt'
        elif data.skew() < -1:
            return 'square'
        else:
            return 'none'
    
    def _classify_skewness(self, skewness):
        """Classify skewness level"""
        if abs(skewness) <= 0.5:
            return 'normal'
        elif abs(skewness) <= 1:
            return 'moderate'
        else:
            return 'high'
    
    def _recommend_scaling_method(self, data):
        """Recommend scaling method for multiple features"""
        has_outliers = any(self._detect_outliers_iqr(data[col].dropna()) for col in data.columns if data[col].count() > 0)
        
        if has_outliers:
            return 'robust'
        elif all(data[col].min() >= 0 for col in data.columns if data[col].count() > 0):
            return 'min_max'
        else:
            return 'standard'
    
    def _estimate_encoding_memory_impact(self, data):
        """Estimate memory impact of encoding"""
        unique_count = data.nunique()
        
        if unique_count <= 5:
            return 'low'
        elif unique_count <= 20:
            return 'medium'
        else:
            return 'high' 