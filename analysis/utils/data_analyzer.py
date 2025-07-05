"""
Data Analyzer Module

Provides comprehensive data analysis capabilities including statistical analysis,
data quality assessment, and pattern identification for business intelligence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Comprehensive data analyzer for business intelligence and statistical analysis
    """
    
    def __init__(self):
        self.analysis_cache = {}
        
    def comprehensive_analysis(self, df):
        """
        Perform comprehensive analysis of the dataset
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            analysis = {
                'basic_statistics': self._get_basic_statistics(df),
                'data_quality': self._assess_data_quality(df),
                'column_analysis': self._analyze_columns(df),
                'correlation_analysis': self._analyze_correlations(df),
                'distribution_analysis': self._analyze_distributions(df),
                'time_series_analysis': self._analyze_time_series(df),
                'categorical_analysis': self._analyze_categorical_data(df),
                'outlier_analysis': self._analyze_outliers(df),
                'business_metrics': self._calculate_business_metrics(df),
                'data_patterns': self._identify_patterns(df)
            }
            
            return self._convert_numpy_types(analysis)
            
        except Exception as e:
            logger.error(f"Error in comprehensive_analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_basic_statistics(self, df):
        """Get basic statistical information about the dataset"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            stats = {
                'dataset_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numerical_columns': len(numerical_cols),
                    'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                    'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                'numerical_summary': {},
                'missing_data': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Numerical statistics
            if len(numerical_cols) > 0:
                numerical_stats = df[numerical_cols].describe()
                stats['numerical_summary'] = numerical_stats.to_dict()
                
                # Additional statistics
                for col in numerical_cols:
                    if df[col].count() > 0:
                        stats['numerical_summary'][col].update({
                            'skewness': df[col].skew(),
                            'kurtosis': df[col].kurtosis(),
                            'variance': df[col].var(),
                            'range': df[col].max() - df[col].min(),
                            'coefficient_of_variation': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in _get_basic_statistics: {str(e)}")
            return {'error': str(e)}
    
    def _assess_data_quality(self, df):
        """Assess data quality issues"""
        try:
            quality_assessment = {
                'completeness': {
                    'total_missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                    'missing_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'duplicates': {
                    'total_duplicates': df.duplicated().sum(),
                    'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                    'unique_rows': len(df) - df.duplicated().sum()
                },
                'consistency': self._check_consistency(df),
                'data_types_issues': self._identify_data_type_issues(df)
            }
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error in _assess_data_quality: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_columns(self, df):
        """Analyze individual columns for patterns and characteristics"""
        try:
            column_analysis = {}
            
            for col in df.columns:
                col_data = df[col].dropna()
                
                analysis = {
                    'data_type': str(df[col].dtype),
                    'unique_values': df[col].nunique(),
                    'unique_ratio': df[col].nunique() / len(df),
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
                }
                
                # Type-specific analysis
                if df[col].dtype in ['int64', 'float64']:
                    analysis.update(self._analyze_numerical_column(col_data))
                elif df[col].dtype == 'object':
                    analysis.update(self._analyze_text_column(col_data))
                elif df[col].dtype.name.startswith('datetime'):
                    analysis.update(self._analyze_datetime_column(col_data))
                
                column_analysis[col] = analysis
            
            return column_analysis
            
        except Exception as e:
            logger.error(f"Error in _analyze_columns: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_numerical_column(self, series):
        """Analyze numerical column"""
        try:
            if len(series) == 0:
                return {}
                
            return {
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'range': series.max() - series.min(),
                'outliers': self._detect_outliers_iqr(series),
                'distribution_type': self._identify_distribution_type(series),
                'is_normal': self._test_normality(series),
                'zero_values': (series == 0).sum(),
                'negative_values': (series < 0).sum(),
                'positive_values': (series > 0).sum()
            }
            
        except Exception as e:
            logger.error(f"Error in _analyze_numerical_column: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_text_column(self, series):
        """Analyze text/categorical column"""
        try:
            if len(series) == 0:
                return {}
                
            value_counts = series.value_counts()
            
            return {
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'cardinality': len(value_counts),
                'is_high_cardinality': len(value_counts) > len(series) * 0.5,
                'is_binary': len(value_counts) == 2,
                'average_length': series.astype(str).str.len().mean(),
                'max_length': series.astype(str).str.len().max(),
                'min_length': series.astype(str).str.len().min(),
                'contains_numbers': series.astype(str).str.contains(r'\d').sum(),
                'contains_special_chars': series.astype(str).str.contains(r'[^a-zA-Z0-9\s]').sum()
            }
            
        except Exception as e:
            logger.error(f"Error in _analyze_text_column: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_datetime_column(self, series):
        """Analyze datetime column"""
        try:
            if len(series) == 0:
                return {}
                
            return {
                'earliest_date': series.min(),
                'latest_date': series.max(),
                'date_range_days': (series.max() - series.min()).days,
                'unique_dates': series.nunique(),
                'most_common_date': series.mode().iloc[0] if len(series.mode()) > 0 else None,
                'year_range': [series.dt.year.min(), series.dt.year.max()],
                'month_distribution': series.dt.month.value_counts().to_dict(),
                'day_of_week_distribution': series.dt.dayofweek.value_counts().to_dict(),
                'has_time_component': (series.dt.hour.nunique() > 1) or (series.dt.minute.nunique() > 1)
            }
            
        except Exception as e:
            logger.error(f"Error in _analyze_datetime_column: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numerical columns"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return {'message': 'Insufficient numerical columns for correlation analysis'}
            
            correlation_matrix = df[numerical_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'highest_correlation': max(strong_correlations, key=lambda x: abs(x['correlation'])) if strong_correlations else None
            }
            
        except Exception as e:
            logger.error(f"Error in _analyze_correlations: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_distributions(self, df):
        """Analyze data distributions"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            distribution_analysis = {}
            
            for col in numerical_cols:
                if df[col].count() > 0:
                    col_data = df[col].dropna()
                    
                    distribution_analysis[col] = {
                        'distribution_type': self._identify_distribution_type(col_data),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'is_normal': self._test_normality(col_data),
                        'outlier_percentage': len(self._detect_outliers_iqr(col_data)) / len(col_data) * 100,
                        'quartiles': {
                            'q1': col_data.quantile(0.25),
                            'q2': col_data.quantile(0.5),
                            'q3': col_data.quantile(0.75)
                        }
                    }
            
            return distribution_analysis
            
        except Exception as e:
            logger.error(f"Error in _analyze_distributions: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_time_series(self, df):
        """Analyze time series patterns if datetime columns exist"""
        try:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return {'message': 'No datetime columns found for time series analysis'}
            
            time_series_analysis = {}
            
            for col in datetime_cols:
                if df[col].count() > 0:
                    # Basic time series info
                    time_series_analysis[col] = {
                        'date_range': {
                            'start': df[col].min(),
                            'end': df[col].max(),
                            'duration_days': (df[col].max() - df[col].min()).days
                        },
                        'frequency_analysis': self._analyze_frequency(df[col]),
                        'seasonal_patterns': self._identify_seasonal_patterns(df, col),
                        'trend_analysis': self._analyze_trend(df, col)
                    }
            
            return time_series_analysis
            
        except Exception as e:
            logger.error(f"Error in _analyze_time_series: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_categorical_data(self, df):
        """Analyze categorical data patterns"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            categorical_analysis = {}
            
            for col in categorical_cols:
                if df[col].count() > 0:
                    value_counts = df[col].value_counts()
                    
                    categorical_analysis[col] = {
                        'unique_categories': len(value_counts),
                        'most_frequent_category': value_counts.index[0],
                        'most_frequent_count': value_counts.iloc[0],
                        'least_frequent_category': value_counts.index[-1],
                        'least_frequent_count': value_counts.iloc[-1],
                        'category_distribution': value_counts.to_dict(),
                        'concentration_ratio': value_counts.iloc[0] / len(df),  # How concentrated is the top category
                        'diversity_index': self._calculate_diversity_index(value_counts)
                    }
            
            return categorical_analysis
            
        except Exception as e:
            logger.error(f"Error in _analyze_categorical_data: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_outliers(self, df):
        """Analyze outliers in numerical columns"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_analysis = {}
            
            for col in numerical_cols:
                if df[col].count() > 0:
                    col_data = df[col].dropna()
                    outliers = self._detect_outliers_iqr(col_data)
                    
                    outlier_analysis[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': len(outliers) / len(col_data) * 100,
                        'outlier_values': outliers.tolist() if len(outliers) < 20 else outliers[:20].tolist(),
                        'outlier_method': 'IQR',
                        'outlier_bounds': {
                            'lower': col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)),
                            'upper': col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))
                        }
                    }
            
            return outlier_analysis
            
        except Exception as e:
            logger.error(f"Error in _analyze_outliers: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_business_metrics(self, df):
        """Calculate business-relevant metrics"""
        try:
            metrics = {
                'data_completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'data_uniqueness_score': (1 - df.duplicated().sum() / len(df)) * 100,
                'data_consistency_score': self._calculate_consistency_score(df),
                'data_freshness': self._calculate_data_freshness(df),
                'data_density': self._calculate_data_density(df)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in _calculate_business_metrics: {str(e)}")
            return {'error': str(e)}
    
    def _identify_patterns(self, df):
        """Identify interesting patterns in the data"""
        try:
            patterns = {
                'growth_patterns': self._identify_growth_patterns(df),
                'seasonal_patterns': self._identify_seasonal_patterns_general(df),
                'anomaly_patterns': self._identify_anomaly_patterns(df),
                'relationship_patterns': self._identify_relationship_patterns(df)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in _identify_patterns: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods
    def _detect_outliers_iqr(self, series):
        """Detect outliers using IQR method"""
        try:
            # Remove NaN values first
            clean_series = series.dropna()
            
            if len(clean_series) < 4:  # Need at least 4 values for IQR
                return pd.Series([], dtype=series.dtype)
            
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # All values are the same
                return pd.Series([], dtype=series.dtype)
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Use proper boolean indexing with explicit conditions
            outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
            outliers = clean_series[outlier_mask]
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return pd.Series([], dtype=series.dtype if hasattr(series, 'dtype') else 'float64')
    
    def _test_normality(self, series):
        """Test if data follows normal distribution"""
        try:
            if len(series) < 8:
                return False
            _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
            return p_value > 0.05
        except:
            return False
    
    def _identify_distribution_type(self, series):
        """Identify the type of distribution"""
        try:
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return 'normal'
            elif skewness > 1:
                return 'right_skewed'
            elif skewness < -1:
                return 'left_skewed'
            elif kurtosis > 3:
                return 'heavy_tailed'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def _check_consistency(self, df):
        """Check data consistency"""
        try:
            consistency_issues = []
            
            # Check for mixed data types in object columns
            for col in df.select_dtypes(include=['object']).columns:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    types = set(type(x).__name__ for x in sample)
                    if len(types) > 1:
                        consistency_issues.append(f"Mixed data types in column '{col}': {types}")
            
            return {
                'issues_found': len(consistency_issues),
                'issues': consistency_issues
            }
        except:
            return {'issues_found': 0, 'issues': []}
    
    def _identify_data_type_issues(self, df):
        """Identify potential data type issues"""
        try:
            issues = []
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if it could be numeric
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        try:
                            pd.to_numeric(sample)
                            issues.append(f"Column '{col}' might be numeric but stored as text")
                        except:
                            pass
                        
                        # Check if it could be datetime
                        try:
                            pd.to_datetime(sample)
                            issues.append(f"Column '{col}' might be datetime but stored as text")
                        except:
                            pass
            
            return issues
        except:
            return []
    
    def _analyze_frequency(self, series):
        """Analyze frequency patterns in datetime series"""
        try:
            if len(series) < 2:
                return {'message': 'Insufficient data for frequency analysis'}
            
            # Calculate time differences
            sorted_series = series.sort_values()
            time_diffs = sorted_series.diff().dropna()
            
            if len(time_diffs) == 0:
                return {'message': 'No time differences found'}
            
            # Find most common frequency
            mode_diff = time_diffs.mode()
            
            return {
                'most_common_interval': str(mode_diff.iloc[0]) if len(mode_diff) > 0 else None,
                'average_interval': str(time_diffs.mean()),
                'irregular_intervals': len(time_diffs.unique()) > 1
            }
        except:
            return {'message': 'Error analyzing frequency'}
    
    def _identify_seasonal_patterns(self, df, datetime_col):
        """Identify seasonal patterns"""
        try:
            # This is a simplified version - in practice, you'd use more sophisticated time series analysis
            df_copy = df.copy()
            df_copy['month'] = df_copy[datetime_col].dt.month
            df_copy['quarter'] = df_copy[datetime_col].dt.quarter
            df_copy['day_of_week'] = df_copy[datetime_col].dt.dayofweek
            
            patterns = {}
            
            # Check for numerical columns to analyze patterns
            numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col not in ['month', 'quarter', 'day_of_week']:
                    patterns[col] = {
                        'monthly_pattern': df_copy.groupby('month')[col].mean().to_dict(),
                        'quarterly_pattern': df_copy.groupby('quarter')[col].mean().to_dict(),
                        'weekly_pattern': df_copy.groupby('day_of_week')[col].mean().to_dict()
                    }
            
            return patterns
        except:
            return {'message': 'Error identifying seasonal patterns'}
    
    def _analyze_trend(self, df, datetime_col):
        """Analyze trend patterns"""
        try:
            # Simple trend analysis
            df_sorted = df.sort_values(datetime_col)
            
            trends = {}
            numerical_cols = df_sorted.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if df_sorted[col].count() > 1:
                    # Calculate simple trend (correlation with time)
                    time_numeric = pd.to_numeric(df_sorted[datetime_col])
                    correlation = df_sorted[col].corr(time_numeric)
                    
                    if abs(correlation) > 0.3:
                        trends[col] = {
                            'trend_direction': 'increasing' if correlation > 0 else 'decreasing',
                            'trend_strength': abs(correlation),
                            'trend_type': 'strong' if abs(correlation) > 0.7 else 'moderate'
                        }
            
            return trends
        except:
            return {'message': 'Error analyzing trends'}
    
    def _calculate_diversity_index(self, value_counts):
        """Calculate diversity index (Shannon entropy)"""
        try:
            proportions = value_counts / value_counts.sum()
            return -sum(p * np.log2(p) for p in proportions if p > 0)
        except:
            return 0
    
    def _calculate_consistency_score(self, df):
        """Calculate overall consistency score"""
        try:
            # Simple consistency score based on data type consistency
            score = 100.0
            
            for col in df.select_dtypes(include=['object']).columns:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    types = set(type(x).__name__ for x in sample)
                    if len(types) > 1:
                        score -= 10  # Penalty for mixed types
            
            return max(0, score)
        except:
            return 0
    
    def _calculate_data_freshness(self, df):
        """Calculate data freshness score"""
        try:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return {'message': 'No datetime columns for freshness analysis'}
            
            # Use the most recent datetime column
            latest_col = datetime_cols[0]
            latest_date = df[latest_col].max()
            current_date = pd.Timestamp.now()
            
            days_old = (current_date - latest_date).days
            
            if days_old < 1:
                return {'score': 100, 'status': 'very_fresh'}
            elif days_old < 7:
                return {'score': 90, 'status': 'fresh'}
            elif days_old < 30:
                return {'score': 70, 'status': 'moderate'}
            else:
                return {'score': 30, 'status': 'stale'}
        except:
            return {'message': 'Error calculating freshness'}
    
    def _calculate_data_density(self, df):
        """Calculate data density (non-null values ratio)"""
        try:
            total_cells = len(df) * len(df.columns)
            non_null_cells = total_cells - df.isnull().sum().sum()
            return (non_null_cells / total_cells) * 100
        except:
            return 0
    
    def _identify_growth_patterns(self, df):
        """Identify growth patterns in the data"""
        try:
            patterns = []
            
            # Look for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                for date_col in datetime_cols:
                    for num_col in numerical_cols:
                        # Group by time periods and calculate growth
                        df_sorted = df.sort_values(date_col)
                        
                        # Monthly growth
                        monthly_data = df_sorted.groupby(df_sorted[date_col].dt.to_period('M'))[num_col].sum()
                        if len(monthly_data) > 1:
                            growth_rate = monthly_data.pct_change().mean()
                            if abs(growth_rate) > 0.05:  # 5% threshold
                                patterns.append({
                                    'type': 'monthly_growth',
                                    'column': num_col,
                                    'growth_rate': growth_rate,
                                    'direction': 'positive' if growth_rate > 0 else 'negative'
                                })
            
            return patterns
        except:
            return []
    
    def _identify_seasonal_patterns_general(self, df):
        """Identify general seasonal patterns"""
        try:
            patterns = []
            
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            for date_col in datetime_cols:
                # Extract seasonal components
                df_temp = df.copy()
                df_temp['month'] = df_temp[date_col].dt.month
                df_temp['quarter'] = df_temp[date_col].dt.quarter
                
                # Check if data shows seasonal patterns
                monthly_counts = df_temp['month'].value_counts()
                if monthly_counts.std() > monthly_counts.mean() * 0.3:
                    patterns.append({
                        'type': 'seasonal',
                        'column': date_col,
                        'pattern': 'monthly_variation',
                        'peak_month': monthly_counts.idxmax(),
                        'low_month': monthly_counts.idxmin()
                    })
            
            return patterns
        except:
            return []
    
    def _identify_anomaly_patterns(self, df):
        """Identify anomaly patterns"""
        try:
            patterns = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if df[col].count() > 0:
                    outliers = self._detect_outliers_iqr(df[col])
                    if len(outliers) > 0:
                        patterns.append({
                            'type': 'outliers',
                            'column': col,
                            'outlier_count': len(outliers),
                            'outlier_percentage': len(outliers) / df[col].count() * 100
                        })
            
            return patterns
        except:
            return []
    
    def _identify_relationship_patterns(self, df):
        """Identify relationship patterns between columns"""
        try:
            patterns = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) >= 2:
                correlation_matrix = df[numerical_cols].corr()
                
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            patterns.append({
                                'type': 'correlation',
                                'column1': numerical_cols[i],
                                'column2': numerical_cols[j],
                                'correlation': corr_value,
                                'relationship': 'positive' if corr_value > 0 else 'negative'
                            })
            
            return patterns
        except:
            return []
    
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
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj 