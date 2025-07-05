"""
Data Analyzer Module

This module provides comprehensive data analysis capabilities for different types of analytics:
- Statistical analysis and data profiling
- Data quality assessment and validation
- Pattern detection and anomaly identification
- Time series analysis
- Outlier detection and handling

Supports descriptive, diagnostic, predictive, and prescriptive analytics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Comprehensive data analyzer for business analytics
    """
    
    def __init__(self):
        """Initialize the data analyzer"""
        self.scaler = StandardScaler()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def comprehensive_analysis(self, df, analysis_type='all'):
        """
        Perform comprehensive analysis on a dataset
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            analysis_type (str): Type of analysis to perform
                - 'all': All types of analysis
                - 'descriptive': Descriptive statistics
                - 'diagnostic': Diagnostic analysis
                - 'predictive': Predictive analysis
                - 'prescriptive': Prescriptive analysis
        
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': self._get_dataset_info(df),
                'data_quality': self._assess_data_quality(df),
                'basic_statistics': self._get_basic_statistics(df)
            }
            
            # Perform specific analysis based on type
            if analysis_type in ['all', 'descriptive']:
                analysis_results['descriptive_analysis'] = self._perform_descriptive_analysis(df)
            
            if analysis_type in ['all', 'diagnostic']:
                analysis_results['diagnostic_analysis'] = self._perform_diagnostic_analysis(df)
            
            if analysis_type in ['all', 'predictive']:
                analysis_results['predictive_analysis'] = self._perform_predictive_analysis(df)
            
            if analysis_type in ['all', 'prescriptive']:
                analysis_results['prescriptive_analysis'] = self._perform_prescriptive_analysis(df)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_dataset_info(self, df):
        """Get basic information about the dataset"""
        try:
            return {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
                'boolean_columns': list(df.select_dtypes(include=['bool']).columns)
            }
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {}
    
    def _assess_data_quality(self, df):
        """Assess data quality issues"""
        try:
            quality_assessment = {
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
                'unique_values': df.nunique().to_dict(),
                'data_types_issues': []
            }
            
            # Check for data type issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if numeric data is stored as string
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        quality_assessment['data_types_issues'].append({
                            'column': col,
                            'issue': 'Numeric data stored as string',
                            'suggestion': 'Convert to numeric'
                        })
                    except:
                        # Check if datetime data is stored as string
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            quality_assessment['data_types_issues'].append({
                                'column': col,
                                'issue': 'Datetime data stored as string',
                                'suggestion': 'Convert to datetime'
                            })
                        except:
                            pass
            
            # Check for outliers in numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numerical_cols:
                if not df[col].empty:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_info[col] = {
                        'count': outliers,
                        'percentage': outliers / len(df) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
            
            quality_assessment['outliers'] = outlier_info
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {}
    
    def _get_basic_statistics(self, df):
        """Get basic statistical information"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            stats_info = {
                'numerical_stats': {},
                'categorical_stats': {}
            }
            
            # Numerical statistics
            if len(numerical_cols) > 0:
                stats_info['numerical_stats'] = df[numerical_cols].describe().to_dict()
                
                # Additional statistics
                for col in numerical_cols:
                    if not df[col].empty:
                        stats_info['numerical_stats'][col].update({
                            'skewness': df[col].skew(),
                            'kurtosis': df[col].kurtosis(),
                            'variance': df[col].var(),
                            'coefficient_of_variation': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        })
            
            # Categorical statistics
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    if not df[col].empty:
                        value_counts = df[col].value_counts()
                        stats_info['categorical_stats'][col] = {
                            'unique_count': df[col].nunique(),
                            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                            'top_5_values': value_counts.head(5).to_dict()
                        }
            
            return stats_info
            
        except Exception as e:
            logger.error(f"Error getting basic statistics: {str(e)}")
            return {}
    
    def _perform_descriptive_analysis(self, df):
        """Perform descriptive analysis"""
        try:
            descriptive_results = {
                'summary_statistics': {},
                'distribution_analysis': {},
                'correlation_analysis': {}
            }
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Summary statistics
            if len(numerical_cols) > 0:
                descriptive_results['summary_statistics'] = {
                    'central_tendency': {
                        'mean': df[numerical_cols].mean().to_dict(),
                        'median': df[numerical_cols].median().to_dict(),
                        'mode': df[numerical_cols].mode().iloc[0].to_dict() if not df[numerical_cols].mode().empty else {}
                    },
                    'dispersion': {
                        'std': df[numerical_cols].std().to_dict(),
                        'var': df[numerical_cols].var().to_dict(),
                        'range': (df[numerical_cols].max() - df[numerical_cols].min()).to_dict()
                    }
                }
            
            # Distribution analysis
            for col in numerical_cols:
                if not df[col].empty:
                    descriptive_results['distribution_analysis'][col] = {
                        'normality_test': self._test_normality(df[col]),
                        'distribution_type': self._identify_distribution_type(df[col]),
                        'skewness_interpretation': self._interpret_skewness(df[col].skew()),
                        'kurtosis_interpretation': self._interpret_kurtosis(df[col].kurtosis())
                    }
            
            # Correlation analysis
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                descriptive_results['correlation_analysis'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'strong_correlations': self._find_strong_correlations(corr_matrix),
                    'correlation_insights': self._generate_correlation_insights(corr_matrix)
                }
            
            return descriptive_results
            
        except Exception as e:
            logger.error(f"Error in descriptive analysis: {str(e)}")
            return {}
    
    def _perform_diagnostic_analysis(self, df):
        """Perform diagnostic analysis"""
        try:
            diagnostic_results = {
                'anomaly_detection': {},
                'pattern_analysis': {},
                'relationship_analysis': {}
            }
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Anomaly detection
            if len(numerical_cols) > 0:
                diagnostic_results['anomaly_detection'] = self._detect_anomalies(df[numerical_cols])
            
            # Pattern analysis
            diagnostic_results['pattern_analysis'] = self._analyze_patterns(df)
            
            # Relationship analysis
            if len(numerical_cols) > 1:
                diagnostic_results['relationship_analysis'] = self._analyze_relationships(df)
            
            return diagnostic_results
            
        except Exception as e:
            logger.error(f"Error in diagnostic analysis: {str(e)}")
            return {}
    
    def _perform_predictive_analysis(self, df):
        """Perform predictive analysis"""
        try:
            predictive_results = {
                'trend_analysis': {},
                'seasonality_analysis': {},
                'forecasting_insights': {}
            }
            
            # Time series analysis if datetime columns exist
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                predictive_results['time_series_analysis'] = self._analyze_time_series(df)
            
            # Trend analysis
            if len(numerical_cols) > 0:
                predictive_results['trend_analysis'] = self._analyze_trends(df)
            
            # Predictive insights
            predictive_results['forecasting_insights'] = self._generate_forecasting_insights(df)
            
            return predictive_results
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {str(e)}")
            return {}
    
    def _perform_prescriptive_analysis(self, df):
        """Perform prescriptive analysis"""
        try:
            prescriptive_results = {
                'optimization_opportunities': {},
                'recommendations': {},
                'business_metrics': {}
            }
            
            # Business metrics calculation
            prescriptive_results['business_metrics'] = self._calculate_business_metrics(df)
            
            # Optimization opportunities
            prescriptive_results['optimization_opportunities'] = self._identify_optimization_opportunities(df)
            
            # Generate recommendations
            prescriptive_results['recommendations'] = self._generate_recommendations(df)
            
            return prescriptive_results
            
        except Exception as e:
            logger.error(f"Error in prescriptive analysis: {str(e)}")
            return {}
    
    def _test_normality(self, series):
        """Test if a series follows normal distribution"""
        try:
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(series.dropna().sample(min(5000, len(series))))
            return {
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'interpretation': 'Normal distribution' if p_value > 0.05 else 'Not normal distribution'
            }
        except Exception as e:
            logger.error(f"Error testing normality: {str(e)}")
            return {}
    
    def _identify_distribution_type(self, series):
        """Identify the type of distribution"""
        try:
            data = series.dropna()
            
            # Test for different distributions
            distributions = {
                'normal': stats.normaltest(data)[1] > 0.05,
                'uniform': stats.kstest(data, 'uniform')[1] > 0.05,
                'exponential': stats.kstest(data, 'expon')[1] > 0.05
            }
            
            # Find the best fitting distribution
            best_fit = max(distributions, key=distributions.get)
            return {
                'best_fit': best_fit,
                'confidence': distributions[best_fit],
                'all_tests': distributions
            }
            
        except Exception as e:
            logger.error(f"Error identifying distribution type: {str(e)}")
            return {}
    
    def _interpret_skewness(self, skewness):
        """Interpret skewness value"""
        if skewness > 1:
            return "Highly right-skewed"
        elif skewness > 0.5:
            return "Moderately right-skewed"
        elif skewness > -0.5:
            return "Approximately symmetric"
        elif skewness > -1:
            return "Moderately left-skewed"
        else:
            return "Highly left-skewed"
    
    def _interpret_kurtosis(self, kurtosis):
        """Interpret kurtosis value"""
        if kurtosis > 3:
            return "Heavy-tailed (leptokurtic)"
        elif kurtosis < -3:
            return "Light-tailed (platykurtic)"
        else:
            return "Normal-tailed (mesokurtic)"
    
    def _find_strong_correlations(self, corr_matrix, threshold=0.7):
        """Find strong correlations in the correlation matrix"""
        try:
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'Strong positive' if corr_value > 0 else 'Strong negative'
                        })
            
            return strong_correlations
            
        except Exception as e:
            logger.error(f"Error finding strong correlations: {str(e)}")
            return []
    
    def _generate_correlation_insights(self, corr_matrix):
        """Generate insights from correlation analysis"""
        try:
            insights = []
            
            # Find the strongest positive and negative correlations
            corr_values = corr_matrix.values
            np.fill_diagonal(corr_values, 0)  # Remove self-correlations
            
            max_corr_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
            min_corr_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
            
            max_corr = corr_values[max_corr_idx]
            min_corr = corr_values[min_corr_idx]
            
            if max_corr > 0.5:
                var1 = corr_matrix.index[max_corr_idx[0]]
                var2 = corr_matrix.columns[max_corr_idx[1]]
                insights.append(f"Strongest positive correlation: {var1} & {var2} ({max_corr:.3f})")
            
            if min_corr < -0.5:
                var1 = corr_matrix.index[min_corr_idx[0]]
                var2 = corr_matrix.columns[min_corr_idx[1]]
                insights.append(f"Strongest negative correlation: {var1} & {var2} ({min_corr:.3f})")
            
            # Overall correlation strength
            avg_abs_corr = np.mean(np.abs(corr_values))
            insights.append(f"Average absolute correlation: {avg_abs_corr:.3f}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating correlation insights: {str(e)}")
            return []
    
    def _detect_anomalies(self, df):
        """Detect anomalies in the dataset"""
        try:
            anomaly_results = {}
            
            # Statistical outlier detection
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = (z_scores > 3).sum()
                    
                    anomaly_results[col] = {
                        'statistical_outliers': outliers,
                        'outlier_percentage': outliers / len(df) * 100
                    }
            
            # Isolation Forest for multivariate anomaly detection
            if len(df.columns) > 1:
                try:
                    # Prepare data for isolation forest
                    data_for_isolation = df.select_dtypes(include=[np.number]).dropna()
                    
                    if not data_for_isolation.empty:
                        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                        anomaly_labels = isolation_forest.fit_predict(data_for_isolation)
                        
                        anomaly_results['multivariate_anomalies'] = {
                            'total_anomalies': (anomaly_labels == -1).sum(),
                            'anomaly_percentage': (anomaly_labels == -1).sum() / len(anomaly_labels) * 100
                        }
                except Exception as e:
                    logger.error(f"Error in isolation forest: {str(e)}")
            
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {}
    
    def _analyze_patterns(self, df):
        """Analyze patterns in the data"""
        try:
            pattern_results = {}
            
            # Analyze categorical patterns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if not df[col].empty:
                    value_counts = df[col].value_counts()
                    pattern_results[col] = {
                        'distribution_type': self._classify_categorical_distribution(value_counts),
                        'concentration_ratio': value_counts.iloc[0] / value_counts.sum(),
                        'diversity_index': self._calculate_diversity_index(value_counts)
                    }
            
            # Analyze numerical patterns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if not df[col].empty:
                    pattern_results[col] = {
                        'trend': self._detect_trend(df[col]),
                        'seasonality': self._detect_seasonality(df[col]),
                        'volatility': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    }
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return {}
    
    def _analyze_relationships(self, df):
        """Analyze relationships between variables"""
        try:
            relationship_results = {}
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 1:
                # Correlation analysis
                corr_matrix = df[numerical_cols].corr()
                
                # Find variable pairs with interesting relationships
                interesting_pairs = []
                
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        
                        if abs(corr_value) > 0.3:  # Threshold for interesting correlation
                            interesting_pairs.append({
                                'variable1': numerical_cols[i],
                                'variable2': numerical_cols[j],
                                'correlation': corr_value,
                                'relationship_type': self._classify_relationship(corr_value)
                            })
                
                relationship_results['interesting_pairs'] = interesting_pairs
                
                # Cluster analysis if applicable
                if len(numerical_cols) >= 3:
                    relationship_results['cluster_analysis'] = self._perform_cluster_analysis(df[numerical_cols])
            
            return relationship_results
            
        except Exception as e:
            logger.error(f"Error analyzing relationships: {str(e)}")
            return {}
    
    def _analyze_time_series(self, df):
        """Analyze time series patterns"""
        try:
            time_series_results = {}
            
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                datetime_col = datetime_cols[0]  # Use first datetime column
                
                # Sort by datetime
                df_sorted = df.sort_values(datetime_col)
                
                for num_col in numerical_cols:
                    if not df_sorted[num_col].empty:
                        time_series_results[num_col] = {
                            'trend': self._analyze_time_trend(df_sorted[datetime_col], df_sorted[num_col]),
                            'seasonality': self._analyze_seasonality(df_sorted[datetime_col], df_sorted[num_col]),
                            'volatility': self._calculate_volatility(df_sorted[num_col]),
                            'stationarity': self._test_stationarity(df_sorted[num_col])
                        }
            
            return time_series_results
            
        except Exception as e:
            logger.error(f"Error analyzing time series: {str(e)}")
            return {}
    
    def _analyze_trends(self, df):
        """Analyze trends in the data"""
        try:
            trend_results = {}
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if not df[col].empty:
                    # Calculate trend using linear regression
                    x = np.arange(len(df[col]))
                    y = df[col].values
                    
                    # Remove NaN values
                    mask = ~np.isnan(y)
                    if mask.sum() > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        
                        trend_results[col] = {
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable',
                            'trend_strength': 'Strong' if abs(r_value) > 0.7 else 'Moderate' if abs(r_value) > 0.3 else 'Weak'
                        }
            
            return trend_results
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {}
    
    def _generate_forecasting_insights(self, df):
        """Generate insights for forecasting"""
        try:
            insights = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Check for time series data
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                insights.append("Time series data detected - suitable for time-based forecasting")
                
                # Check for seasonality patterns
                for col in numerical_cols:
                    if not df[col].empty:
                        cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        if cv > 0.3:
                            insights.append(f"High variability in {col} - consider seasonal models")
                        else:
                            insights.append(f"Low variability in {col} - linear models may work well")
            
            # Check for trend patterns
            for col in numerical_cols:
                if not df[col].empty:
                    # Simple trend detection
                    x = np.arange(len(df[col]))
                    y = df[col].values
                    mask = ~np.isnan(y)
                    
                    if mask.sum() > 1:
                        slope, _, r_value, _, _ = stats.linregress(x[mask], y[mask])
                        
                        if abs(r_value) > 0.5:
                            direction = "upward" if slope > 0 else "downward"
                            insights.append(f"Strong {direction} trend in {col} - trend models recommended")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating forecasting insights: {str(e)}")
            return []
    
    def _calculate_business_metrics(self, df):
        """Calculate business-relevant metrics"""
        try:
            metrics = {}
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Calculate key business metrics
            for col in numerical_cols:
                if not df[col].empty:
                    metrics[col] = {
                        'total': df[col].sum(),
                        'average': df[col].mean(),
                        'growth_rate': self._calculate_growth_rate(df[col]),
                        'consistency': 1 - (df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0,
                        'efficiency_score': self._calculate_efficiency_score(df[col])
                    }
            
            # Calculate derived metrics
            if len(numerical_cols) >= 2:
                metrics['cross_metrics'] = {
                    'ratio_analysis': self._calculate_ratio_analysis(df[numerical_cols]),
                    'performance_comparison': self._compare_performance(df[numerical_cols])
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {str(e)}")
            return {}
    
    def _identify_optimization_opportunities(self, df):
        """Identify optimization opportunities"""
        try:
            opportunities = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Identify outliers as optimization opportunities
            for col in numerical_cols:
                if not df[col].empty:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                    
                    if outliers > 0:
                        opportunities.append({
                            'type': 'Outlier Management',
                            'column': col,
                            'description': f"Found {outliers} outliers in {col}",
                            'potential_impact': 'High' if outliers > len(df) * 0.05 else 'Medium',
                            'recommendation': 'Investigate and potentially remove or transform outliers'
                        })
            
            # Identify missing value opportunities
            missing_values = df.isnull().sum()
            for col, missing_count in missing_values.items():
                if missing_count > 0:
                    opportunities.append({
                        'type': 'Data Quality',
                        'column': col,
                        'description': f"Missing {missing_count} values in {col}",
                        'potential_impact': 'High' if missing_count > len(df) * 0.1 else 'Medium',
                        'recommendation': 'Implement data collection or imputation strategy'
                    })
            
            # Identify correlation opportunities
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                
                # Find highly correlated variables
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        
                        if abs(corr_value) > 0.8:
                            opportunities.append({
                                'type': 'Feature Engineering',
                                'columns': [numerical_cols[i], numerical_cols[j]],
                                'description': f"High correlation ({corr_value:.3f}) between variables",
                                'potential_impact': 'Medium',
                                'recommendation': 'Consider feature reduction or combination'
                            })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
    
    def _generate_recommendations(self, df):
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Data quality recommendations
            missing_percentage = df.isnull().sum() / len(df) * 100
            high_missing = missing_percentage[missing_percentage > 10]
            
            if len(high_missing) > 0:
                recommendations.append({
                    'category': 'Data Quality',
                    'priority': 'High',
                    'recommendation': f"Address missing values in {list(high_missing.index)}",
                    'rationale': 'High missing value rates can significantly impact analysis quality'
                })
            
            # Performance recommendations
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if not df[col].empty:
                    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    
                    if cv > 0.5:
                        recommendations.append({
                            'category': 'Performance',
                            'priority': 'Medium',
                            'recommendation': f"Investigate high variability in {col}",
                            'rationale': 'High variability may indicate process inconsistencies'
                        })
            
            # Business process recommendations
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if not df[col].empty:
                    unique_ratio = df[col].nunique() / len(df)
                    
                    if unique_ratio > 0.8:
                        recommendations.append({
                            'category': 'Process Optimization',
                            'priority': 'Low',
                            'recommendation': f"Consider standardizing {col} values",
                            'rationale': 'High unique value ratio suggests potential categorization opportunities'
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    # Helper methods
    def _classify_categorical_distribution(self, value_counts):
        """Classify the distribution of categorical data"""
        total = value_counts.sum()
        top_percentage = value_counts.iloc[0] / total
        
        if top_percentage > 0.8:
            return "Highly concentrated"
        elif top_percentage > 0.5:
            return "Moderately concentrated"
        else:
            return "Well distributed"
    
    def _calculate_diversity_index(self, value_counts):
        """Calculate Shannon diversity index"""
        proportions = value_counts / value_counts.sum()
        return -sum(p * np.log(p) for p in proportions if p > 0)
    
    def _detect_trend(self, series):
        """Detect trend in a series"""
        x = np.arange(len(series))
        y = series.values
        mask = ~np.isnan(y)
        
        if mask.sum() > 1:
            slope, _, r_value, _, _ = stats.linregress(x[mask], y[mask])
            return {
                'direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable',
                'strength': abs(r_value),
                'slope': slope
            }
        return {}
    
    def _detect_seasonality(self, series):
        """Detect seasonality in a series"""
        # Simple seasonality detection using autocorrelation
        try:
            if len(series) > 10:
                autocorr = [series.autocorr(lag=i) for i in range(1, min(len(series)//4, 50))]
                max_autocorr = max(autocorr) if autocorr else 0
                
                return {
                    'seasonal': max_autocorr > 0.3,
                    'strength': max_autocorr,
                    'period': autocorr.index(max_autocorr) + 1 if max_autocorr > 0.3 else None
                }
        except:
            pass
        
        return {'seasonal': False, 'strength': 0}
    
    def _classify_relationship(self, correlation):
        """Classify relationship based on correlation"""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.7:
            strength = "Strong"
        elif abs_corr > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return f"{strength} {direction}"
    
    def _perform_cluster_analysis(self, df):
        """Perform basic cluster analysis"""
        try:
            # Use DBSCAN for cluster analysis
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df.dropna())
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_data)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            return {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_quality': 'Good' if n_clusters > 1 and n_noise < len(cluster_labels) * 0.1 else 'Poor'
            }
            
        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            return {}
    
    def _analyze_time_trend(self, datetime_col, value_col):
        """Analyze trend over time"""
        try:
            # Convert datetime to numeric for regression
            time_numeric = pd.to_numeric(datetime_col)
            
            mask = ~(pd.isna(time_numeric) | pd.isna(value_col))
            
            if mask.sum() > 1:
                slope, _, r_value, _, _ = stats.linregress(time_numeric[mask], value_col[mask])
                
                return {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable'
                }
        except Exception as e:
            logger.error(f"Error analyzing time trend: {str(e)}")
        
        return {}
    
    def _analyze_seasonality(self, datetime_col, value_col):
        """Analyze seasonality patterns"""
        try:
            # Extract time components
            df_temp = pd.DataFrame({
                'datetime': datetime_col,
                'value': value_col
            }).dropna()
            
            df_temp['month'] = df_temp['datetime'].dt.month
            df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
            df_temp['hour'] = df_temp['datetime'].dt.hour
            
            # Analyze monthly patterns
            monthly_stats = df_temp.groupby('month')['value'].agg(['mean', 'std']).reset_index()
            monthly_cv = monthly_stats['std'].mean() / monthly_stats['mean'].mean()
            
            return {
                'monthly_seasonality': monthly_cv > 0.2,
                'seasonal_strength': monthly_cv,
                'peak_month': monthly_stats.loc[monthly_stats['mean'].idxmax(), 'month']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return {}
    
    def _calculate_volatility(self, series):
        """Calculate volatility of a series"""
        try:
            returns = series.pct_change().dropna()
            return returns.std()
        except:
            return 0
    
    def _test_stationarity(self, series):
        """Test for stationarity"""
        try:
            # Simple stationarity test using rolling statistics
            rolling_mean = series.rolling(window=12).mean()
            rolling_std = series.rolling(window=12).std()
            
            # Check if rolling statistics are relatively stable
            mean_stability = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else float('inf')
            std_stability = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else float('inf')
            
            is_stationary = mean_stability < 0.1 and std_stability < 0.1
            
            return {
                'is_stationary': is_stationary,
                'mean_stability': mean_stability,
                'std_stability': std_stability
            }
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {str(e)}")
            return {}
    
    def _calculate_growth_rate(self, series):
        """Calculate growth rate"""
        try:
            if len(series) > 1:
                first_value = series.iloc[0]
                last_value = series.iloc[-1]
                
                if first_value != 0:
                    return (last_value - first_value) / first_value
            return 0
        except:
            return 0
    
    def _calculate_efficiency_score(self, series):
        """Calculate efficiency score based on consistency"""
        try:
            cv = series.std() / series.mean() if series.mean() != 0 else float('inf')
            return max(0, 1 - cv)  # Higher score for lower variability
        except:
            return 0
    
    def _calculate_ratio_analysis(self, df):
        """Calculate various ratios between numerical columns"""
        try:
            ratios = {}
            cols = df.columns
            
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    col1, col2 = cols[i], cols[j]
                    
                    if df[col2].mean() != 0:
                        ratio = df[col1].mean() / df[col2].mean()
                        ratios[f"{col1}_to_{col2}"] = ratio
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratio analysis: {str(e)}")
            return {}
    
    def _compare_performance(self, df):
        """Compare performance across numerical columns"""
        try:
            performance = {}
            
            for col in df.columns:
                performance[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0,
                    'percentile_75': df[col].quantile(0.75),
                    'percentile_25': df[col].quantile(0.25)
                }
            
            # Rank columns by different metrics
            rankings = {
                'by_mean': sorted(performance.keys(), key=lambda x: performance[x]['mean'], reverse=True),
                'by_consistency': sorted(performance.keys(), key=lambda x: performance[x]['cv']),
                'by_stability': sorted(performance.keys(), key=lambda x: performance[x]['std'])
            }
            
            return {
                'performance_metrics': performance,
                'rankings': rankings
            }
            
        except Exception as e:
            logger.error(f"Error comparing performance: {str(e)}")
            return {} 