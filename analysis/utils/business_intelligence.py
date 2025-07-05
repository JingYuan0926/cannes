"""
Business Intelligence Engine Module

Provides comprehensive business intelligence capabilities including
insights generation, strategic recommendations, and business analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BusinessIntelligenceEngine:
    """
    Comprehensive business intelligence engine for strategic analysis and insights
    """
    
    def __init__(self):
        self.insight_generators = {
            'growth_analysis': self._analyze_growth_patterns,
            'trend_analysis': self._analyze_trends,
            'performance_analysis': self._analyze_performance,
            'customer_analysis': self._analyze_customer_patterns,
            'financial_analysis': self._analyze_financial_metrics,
            'operational_analysis': self._analyze_operational_metrics,
            'market_analysis': self._analyze_market_patterns,
            'risk_analysis': self._analyze_risk_factors,
            'opportunity_analysis': self._analyze_opportunities,
            'competitive_analysis': self._analyze_competitive_position
        }
    
    def generate_business_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                 user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive business insights based on data analysis
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (Dict): Results from data analysis
            user_prompt (str, optional): User's business question or focus area
            
        Returns:
            Dict: Comprehensive business insights
        """
        try:
            insights = {
                'executive_summary': self._generate_executive_summary(df, analysis_results),
                'key_findings': self._generate_key_findings(df, analysis_results),
                'strategic_recommendations': self._generate_strategic_recommendations(df, analysis_results),
                'performance_metrics': self._generate_performance_metrics(df, analysis_results),
                'trend_analysis': self._generate_trend_analysis(df, analysis_results),
                'risk_assessment': self._generate_risk_assessment(df, analysis_results),
                'opportunity_identification': self._generate_opportunity_identification(df, analysis_results),
                'actionable_insights': self._generate_actionable_insights(df, analysis_results),
                'data_quality_assessment': self._generate_data_quality_assessment(df, analysis_results),
                'business_context': self._generate_business_context(df, user_prompt)
            }
            
            # Add user-specific insights if prompt provided
            if user_prompt:
                insights['user_specific_insights'] = self._generate_user_specific_insights(
                    df, analysis_results, user_prompt
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating business insights: {str(e)}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of key business metrics"""
        try:
            summary = {
                'dataset_overview': {
                    'total_records': len(df),
                    'time_period': self._identify_time_period(df),
                    'key_metrics_count': len(df.select_dtypes(include=[np.number]).columns),
                    'data_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                },
                'business_health_score': self._calculate_business_health_score(df, analysis_results),
                'critical_alerts': self._identify_critical_alerts(df, analysis_results),
                'top_opportunities': self._identify_top_opportunities(df, analysis_results)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return {}
    
    def _generate_key_findings(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate key business findings from data"""
        try:
            findings = []
            
            # Statistical findings
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col in df.columns:
                    # Growth patterns
                    if self._is_time_series_column(df, col):
                        growth_rate = self._calculate_growth_rate(df, col)
                        if abs(growth_rate) > 5:  # Significant growth/decline
                            findings.append({
                                'type': 'growth_pattern',
                                'metric': col,
                                'value': growth_rate,
                                'description': f"{col} shows {'growth' if growth_rate > 0 else 'decline'} of {abs(growth_rate):.1f}%",
                                'impact': 'high' if abs(growth_rate) > 20 else 'medium',
                                'recommendation': self._get_growth_recommendation(growth_rate, col)
                            })
                    
                    # Outlier detection
                    outliers = self._detect_outliers(df[col])
                    if len(outliers) > 0:
                        findings.append({
                            'type': 'outlier_detection',
                            'metric': col,
                            'value': len(outliers),
                            'description': f"Detected {len(outliers)} outliers in {col}",
                            'impact': 'medium',
                            'recommendation': f"Investigate outliers in {col} for data quality or exceptional cases"
                        })
            
            # Correlation findings
            if 'correlations' in analysis_results:
                strong_correlations = self._identify_strong_correlations(analysis_results['correlations'])
                for corr in strong_correlations:
                    findings.append({
                        'type': 'correlation',
                        'metrics': [corr['var1'], corr['var2']],
                        'value': corr['correlation'],
                        'description': f"Strong {'positive' if corr['correlation'] > 0 else 'negative'} correlation between {corr['var1']} and {corr['var2']}",
                        'impact': 'high' if abs(corr['correlation']) > 0.8 else 'medium',
                        'recommendation': f"Leverage relationship between {corr['var1']} and {corr['var2']} for strategic decisions"
                    })
            
            # Data quality findings
            missing_data_findings = self._analyze_missing_data_patterns(df)
            findings.extend(missing_data_findings)
            
            return findings
            
        except Exception as e:
            logger.error(f"Error generating key findings: {str(e)}")
            return []
    
    def _generate_strategic_recommendations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic business recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Identify top and bottom performers
            for col in numerical_cols:
                if col in df.columns and len(df[col].dropna()) > 0:
                    performance_insights = self._analyze_performance_distribution(df, col)
                    if performance_insights:
                        recommendations.append({
                            'category': 'performance_optimization',
                            'priority': 'high',
                            'title': f"Optimize {col} Performance",
                            'description': performance_insights['description'],
                            'actions': performance_insights['actions'],
                            'expected_impact': performance_insights['impact'],
                            'timeline': '3-6 months'
                        })
            
            # Growth opportunities
            growth_opportunities = self._identify_growth_opportunities(df, analysis_results)
            recommendations.extend(growth_opportunities)
            
            # Risk mitigation
            risk_recommendations = self._generate_risk_mitigation_recommendations(df, analysis_results)
            recommendations.extend(risk_recommendations)
            
            # Data-driven process improvements
            process_improvements = self._identify_process_improvements(df, analysis_results)
            recommendations.extend(process_improvements)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {str(e)}")
            return []
    
    def _generate_performance_metrics(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate key performance metrics"""
        try:
            metrics = {}
            
            # Calculate KPIs based on data characteristics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if col in df.columns:
                    col_metrics = {
                        'current_value': df[col].iloc[-1] if len(df) > 0 else 0,
                        'average': df[col].mean(),
                        'trend': self._calculate_trend_direction(df, col),
                        'volatility': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0,
                        'percentile_rank': self._calculate_percentile_rank(df[col])
                    }
                    
                    # Add time-based metrics if applicable
                    if self._is_time_series_column(df, col):
                        col_metrics.update({
                            'growth_rate': self._calculate_growth_rate(df, col),
                            'seasonal_pattern': self._detect_seasonality(df, col),
                            'forecast_direction': self._predict_trend_direction(df, col)
                        })
                    
                    metrics[col] = col_metrics
            
            # Business health indicators
            metrics['business_health'] = {
                'data_quality_score': self._calculate_data_quality_score(df),
                'performance_consistency': self._calculate_performance_consistency(df),
                'growth_sustainability': self._assess_growth_sustainability(df),
                'risk_level': self._assess_overall_risk_level(df, analysis_results)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating performance metrics: {str(e)}")
            return {}
    
    def _generate_trend_analysis(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trend analysis"""
        try:
            trend_analysis = {
                'overall_trends': {},
                'seasonal_patterns': {},
                'trend_changes': {},
                'future_projections': {}
            }
            
            # Analyze trends for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if col in df.columns and len(df[col].dropna()) > 2:
                    # Overall trend
                    trend_analysis['overall_trends'][col] = {
                        'direction': self._calculate_trend_direction(df, col),
                        'strength': self._calculate_trend_strength(df, col),
                        'significance': self._test_trend_significance(df, col)
                    }
                    
                    # Seasonal patterns
                    if self._is_time_series_column(df, col):
                        seasonal_info = self._analyze_seasonal_patterns(df, col)
                        if seasonal_info:
                            trend_analysis['seasonal_patterns'][col] = seasonal_info
                    
                    # Trend changes (breakpoints)
                    trend_changes = self._detect_trend_changes(df, col)
                    if trend_changes:
                        trend_analysis['trend_changes'][col] = trend_changes
                    
                    # Future projections
                    projection = self._generate_trend_projection(df, col)
                    if projection:
                        trend_analysis['future_projections'][col] = projection
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            return {}
    
    def _generate_risk_assessment(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        try:
            risk_assessment = {
                'data_quality_risks': self._assess_data_quality_risks(df),
                'performance_risks': self._assess_performance_risks(df, analysis_results),
                'operational_risks': self._assess_operational_risks(df),
                'market_risks': self._assess_market_risks(df),
                'overall_risk_score': 0,
                'mitigation_strategies': []
            }
            
            # Calculate overall risk score
            risk_scores = []
            for risk_category in ['data_quality_risks', 'performance_risks', 'operational_risks', 'market_risks']:
                if risk_category in risk_assessment and 'score' in risk_assessment[risk_category]:
                    risk_scores.append(risk_assessment[risk_category]['score'])
            
            if risk_scores:
                risk_assessment['overall_risk_score'] = np.mean(risk_scores)
            
            # Generate mitigation strategies
            risk_assessment['mitigation_strategies'] = self._generate_mitigation_strategies(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error generating risk assessment: {str(e)}")
            return {}
    
    def _generate_opportunity_identification(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify business opportunities from data"""
        try:
            opportunities = []
            
            # Growth opportunities
            growth_opps = self._identify_growth_opportunities(df, analysis_results)
            opportunities.extend(growth_opps)
            
            # Efficiency opportunities
            efficiency_opps = self._identify_efficiency_opportunities(df, analysis_results)
            opportunities.extend(efficiency_opps)
            
            # Market opportunities
            market_opps = self._identify_market_opportunities(df, analysis_results)
            opportunities.extend(market_opps)
            
            # Innovation opportunities
            innovation_opps = self._identify_innovation_opportunities(df, analysis_results)
            opportunities.extend(innovation_opps)
            
            # Sort by potential impact
            opportunities.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {str(e)}")
            return []
    
    def _generate_actionable_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific actionable insights"""
        try:
            insights = []
            
            # Performance-based actions
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if col in df.columns:
                    # Identify underperforming areas
                    underperformers = self._identify_underperformers(df, col)
                    if underperformers:
                        insights.append({
                            'type': 'performance_improvement',
                            'metric': col,
                            'action': f"Focus on improving {col} in underperforming segments",
                            'specifics': underperformers,
                            'priority': 'high',
                            'timeline': '1-3 months',
                            'expected_outcome': f"Improve overall {col} performance by 15-25%"
                        })
                    
                    # Identify top performers to replicate
                    top_performers = self._identify_top_performers(df, col)
                    if top_performers:
                        insights.append({
                            'type': 'best_practice_replication',
                            'metric': col,
                            'action': f"Replicate success factors from top-performing {col} segments",
                            'specifics': top_performers,
                            'priority': 'medium',
                            'timeline': '2-4 months',
                            'expected_outcome': f"Scale successful {col} practices across organization"
                        })
            
            # Data-driven process improvements
            process_insights = self._generate_process_insights(df, analysis_results)
            insights.extend(process_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating actionable insights: {str(e)}")
            return []
    
    def _generate_data_quality_assessment(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality and its business impact"""
        try:
            assessment = {
                'completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'consistency_score': self._calculate_consistency_score(df),
                'accuracy_indicators': self._assess_accuracy_indicators(df),
                'timeliness_assessment': self._assess_data_timeliness(df),
                'business_impact': self._assess_data_quality_business_impact(df),
                'improvement_recommendations': self._generate_data_quality_recommendations(df)
            }
            
            # Overall data quality score
            scores = [assessment['completeness_score'], assessment['consistency_score']]
            assessment['overall_quality_score'] = np.mean([s for s in scores if s is not None])
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating data quality assessment: {str(e)}")
            return {}
    
    def _generate_business_context(self, df: pd.DataFrame, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate business context and domain insights"""
        try:
            context = {
                'data_characteristics': self._analyze_data_characteristics(df),
                'business_domain': self._infer_business_domain(df),
                'key_business_processes': self._identify_key_processes(df),
                'stakeholder_interests': self._identify_stakeholder_interests(df),
                'competitive_context': self._analyze_competitive_context(df)
            }
            
            if user_prompt:
                context['user_focus_area'] = self._analyze_user_focus_area(user_prompt)
                context['relevant_metrics'] = self._identify_relevant_metrics(df, user_prompt)
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating business context: {str(e)}")
            return {}
    
    def _generate_user_specific_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                       user_prompt: str) -> Dict[str, Any]:
        """Generate insights specific to user's query"""
        try:
            prompt_lower = user_prompt.lower()
            insights = {
                'query_interpretation': self._interpret_user_query(user_prompt),
                'relevant_findings': [],
                'specific_recommendations': [],
                'targeted_metrics': {}
            }
            
            # Identify relevant columns based on user prompt
            relevant_columns = self._identify_relevant_columns(df, user_prompt)
            
            # Generate specific insights for relevant columns
            for col in relevant_columns:
                if col in df.columns:
                    col_insights = self._generate_column_specific_insights(df, col, user_prompt)
                    if col_insights:
                        insights['relevant_findings'].extend(col_insights)
            
            # Generate targeted recommendations
            targeted_recommendations = self._generate_targeted_recommendations(df, user_prompt, analysis_results)
            insights['specific_recommendations'] = targeted_recommendations
            
            # Calculate targeted metrics
            targeted_metrics = self._calculate_targeted_metrics(df, user_prompt)
            insights['targeted_metrics'] = targeted_metrics
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating user-specific insights: {str(e)}")
            return {}
    
    # Helper methods (implementing key functionality)
    def _identify_time_period(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify the time period covered by the dataset"""
        try:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                date_col = datetime_cols[0]
                return {
                    'start_date': df[date_col].min().isoformat() if pd.notnull(df[date_col].min()) else None,
                    'end_date': df[date_col].max().isoformat() if pd.notnull(df[date_col].max()) else None,
                    'duration_days': (df[date_col].max() - df[date_col].min()).days if pd.notnull(df[date_col].min()) else None
                }
            return {'message': 'No datetime columns found'}
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_business_health_score(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall business health score"""
        try:
            scores = []
            
            # Data quality score
            data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            scores.append(data_quality)
            
            # Performance consistency
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                consistency_scores = []
                for col in numerical_cols:
                    if len(df[col].dropna()) > 1:
                        cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        consistency_score = max(0, 100 - (cv * 100))
                        consistency_scores.append(consistency_score)
                if consistency_scores:
                    scores.append(np.mean(consistency_scores))
            
            return np.mean(scores) if scores else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating business health score: {str(e)}")
            return 50.0
    
    def _calculate_growth_rate(self, df: pd.DataFrame, column: str) -> float:
        """Calculate growth rate for a column"""
        try:
            if not pd.api.types.is_numeric_dtype(df[column]):
                return 0.0
                
            # Sort by first datetime column if available
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                df_sorted = df.sort_values(datetime_cols[0])
                values = df_sorted[column].dropna()
            else:
                values = df[column].dropna()
            
            if len(values) < 2:
                return 0.0
                
            # Calculate growth rate between first and last values
            first_value = values.iloc[0]
            last_value = values.iloc[-1]
            
            if first_value == 0:
                return 0.0
                
            growth_rate = ((last_value - first_value) / first_value) * 100
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error calculating growth rate: {str(e)}")
            return 0.0
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        try:
            if not pd.api.types.is_numeric_dtype(series):
                return []
                
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers.index.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return []
    
    def _identify_strong_correlations(self, correlation_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strong correlations from correlation analysis"""
        try:
            strong_correlations = []
            
            if 'correlation_matrix' in correlation_analysis:
                corr_matrix = correlation_analysis['correlation_matrix']
                
                # Convert to DataFrame if it's a dict
                if isinstance(corr_matrix, dict):
                    corr_df = pd.DataFrame(corr_matrix)
                else:
                    corr_df = corr_matrix
                
                # Find strong correlations (> 0.7 or < -0.7)
                for i in range(len(corr_df.columns)):
                    for j in range(i + 1, len(corr_df.columns)):
                        corr_value = corr_df.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append({
                                'var1': corr_df.columns[i],
                                'var2': corr_df.columns[j],
                                'correlation': corr_value
                            })
            
            return strong_correlations
            
        except Exception as e:
            logger.error(f"Error identifying strong correlations: {str(e)}")
            return []
    
    def _is_time_series_column(self, df: pd.DataFrame, column: str) -> bool:
        """Check if a column can be analyzed as time series"""
        try:
            # Check if there's a datetime column in the DataFrame
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            return len(datetime_cols) > 0 and pd.api.types.is_numeric_dtype(df[column])
        except Exception as e:
            return False
    
    def _calculate_trend_direction(self, df: pd.DataFrame, column: str) -> str:
        """Calculate trend direction for a column"""
        try:
            if len(df) < 2:
                return 'stable'
            
            # Simple linear trend
            x = np.arange(len(df))
            y = df[column].values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 'stable'
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            slope, _, _, p_value, _ = stats.linregress(x_clean, y_clean)
            
            if p_value > 0.05:  # Not statistically significant
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {str(e)}")
            return 'stable'
    
    def _interpret_user_query(self, user_prompt: str) -> Dict[str, Any]:
        """Interpret user's business query"""
        try:
            prompt_lower = user_prompt.lower()
            
            # Identify query type
            query_type = 'general'
            if any(word in prompt_lower for word in ['growth', 'increase', 'decrease', 'trend']):
                query_type = 'growth_analysis'
            elif any(word in prompt_lower for word in ['performance', 'kpi', 'metric']):
                query_type = 'performance_analysis'
            elif any(word in prompt_lower for word in ['customer', 'client', 'user']):
                query_type = 'customer_analysis'
            elif any(word in prompt_lower for word in ['revenue', 'profit', 'cost', 'financial']):
                query_type = 'financial_analysis'
            elif any(word in prompt_lower for word in ['risk', 'threat', 'problem']):
                query_type = 'risk_analysis'
            elif any(word in prompt_lower for word in ['opportunity', 'potential', 'improvement']):
                query_type = 'opportunity_analysis'
            
            # Extract time references
            time_references = []
            time_patterns = ['year', 'month', 'quarter', 'week', 'daily', 'annual', 'monthly']
            for pattern in time_patterns:
                if pattern in prompt_lower:
                    time_references.append(pattern)
            
            # Extract metrics mentioned
            mentioned_metrics = []
            # This would be enhanced with more sophisticated NLP
            
            return {
                'query_type': query_type,
                'time_references': time_references,
                'mentioned_metrics': mentioned_metrics,
                'focus_areas': self._extract_focus_areas(prompt_lower)
            }
            
        except Exception as e:
            logger.error(f"Error interpreting user query: {str(e)}")
            return {}
    
    def _extract_focus_areas(self, prompt_lower: str) -> List[str]:
        """Extract focus areas from user prompt"""
        focus_areas = []
        
        business_areas = {
            'sales': ['sales', 'revenue', 'selling'],
            'marketing': ['marketing', 'campaign', 'advertising'],
            'operations': ['operations', 'operational', 'process'],
            'finance': ['finance', 'financial', 'cost', 'profit'],
            'customer': ['customer', 'client', 'user'],
            'product': ['product', 'service', 'offering'],
            'growth': ['growth', 'expansion', 'scale'],
            'efficiency': ['efficiency', 'productivity', 'optimization']
        }
        
        for area, keywords in business_areas.items():
            if any(keyword in prompt_lower for keyword in keywords):
                focus_areas.append(area)
        
        return focus_areas
    
    def _identify_relevant_columns(self, df: pd.DataFrame, user_prompt: str) -> List[str]:
        """Identify columns relevant to user's query"""
        try:
            relevant_columns = []
            prompt_lower = user_prompt.lower()
            
            # Direct column name matches
            for col in df.columns:
                if col.lower() in prompt_lower or col.lower().replace('_', ' ') in prompt_lower:
                    relevant_columns.append(col)
            
            # Semantic matches based on common business terms
            business_terms = {
                'revenue': ['revenue', 'sales', 'income', 'earnings'],
                'cost': ['cost', 'expense', 'spend', 'expenditure'],
                'profit': ['profit', 'margin', 'net', 'earnings'],
                'customer': ['customer', 'client', 'user', 'account'],
                'product': ['product', 'item', 'service', 'sku'],
                'time': ['date', 'time', 'period', 'month', 'year'],
                'quantity': ['quantity', 'volume', 'count', 'number'],
                'rate': ['rate', 'percentage', 'ratio', 'percent']
            }
            
            for term, keywords in business_terms.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    # Find columns that might represent this concept
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in keywords):
                            if col not in relevant_columns:
                                relevant_columns.append(col)
            
            return relevant_columns
            
        except Exception as e:
            logger.error(f"Error identifying relevant columns: {str(e)}")
            return []
    
    # Additional helper methods would be implemented here...
    # For brevity, I'm including key methods that demonstrate the functionality
    
    def _generate_column_specific_insights(self, df: pd.DataFrame, column: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Generate specific insights for a column based on user prompt"""
        try:
            insights = []
            
            if pd.api.types.is_numeric_dtype(df[column]):
                # Statistical insights
                insights.append({
                    'type': 'statistical_summary',
                    'column': column,
                    'description': f"{column} statistics: Mean={df[column].mean():.2f}, Std={df[column].std():.2f}",
                    'business_relevance': self._assess_business_relevance(column, user_prompt)
                })
                
                # Trend insights
                if self._is_time_series_column(df, column):
                    trend = self._calculate_trend_direction(df, column)
                    insights.append({
                        'type': 'trend_analysis',
                        'column': column,
                        'description': f"{column} shows {trend} trend",
                        'business_relevance': f"This trend in {column} is relevant to your query about {user_prompt}"
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating column-specific insights: {str(e)}")
            return []
    
    def _assess_business_relevance(self, column: str, user_prompt: str) -> str:
        """Assess business relevance of a column to user's query"""
        try:
            prompt_lower = user_prompt.lower()
            col_lower = column.lower()
            
            # Simple relevance scoring
            if col_lower in prompt_lower:
                return "Highly relevant - directly mentioned in query"
            elif any(word in col_lower for word in prompt_lower.split()):
                return "Moderately relevant - related terms found"
            else:
                return "General relevance - part of overall analysis"
                
        except Exception as e:
            return "Unable to assess relevance"

    # Missing methods that are being called
    def _identify_critical_alerts(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical business alerts"""
        try:
            alerts = []
            
            # Check for data quality issues
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 20:
                alerts.append({
                    'type': 'data_quality',
                    'severity': 'high',
                    'message': f'High missing data: {missing_percentage:.1f}%',
                    'action': 'Review data collection processes'
                })
            
            # Check for outliers
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                outliers = self._detect_outliers(df[col])
                if len(outliers) > len(df) * 0.1:  # More than 10% outliers
                    alerts.append({
                        'type': 'outliers',
                        'severity': 'medium',
                        'message': f'High outlier count in {col}: {len(outliers)} records',
                        'action': f'Investigate outliers in {col}'
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error identifying critical alerts: {str(e)}")
            return []

    def _identify_top_opportunities(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify top business opportunities"""
        try:
            opportunities = []
            
            # Growth opportunities
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self._is_time_series_column(df, col):
                    growth_rate = self._calculate_growth_rate(df, col)
                    if growth_rate > 10:  # Positive growth
                        opportunities.append({
                            'type': 'growth',
                            'metric': col,
                            'value': growth_rate,
                            'description': f'{col} showing strong growth of {growth_rate:.1f}%',
                            'potential': 'high'
                        })
            
            # Correlation opportunities
            if 'correlation_analysis' in analysis_results:
                strong_correlations = self._identify_strong_correlations(analysis_results['correlation_analysis'])
                for corr in strong_correlations[:3]:  # Top 3
                    opportunities.append({
                        'type': 'correlation',
                        'metrics': [corr['var1'], corr['var2']],
                        'value': corr['correlation'],
                        'description': f'Strong relationship between {corr["var1"]} and {corr["var2"]}',
                        'potential': 'medium'
                    })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {str(e)}")
            return []

    def _get_growth_recommendation(self, growth_rate: float, column: str) -> str:
        """Get recommendation based on growth rate"""
        if growth_rate > 20:
            return f"Capitalize on strong growth in {column} - consider scaling strategies"
        elif growth_rate > 5:
            return f"Monitor and sustain positive growth in {column}"
        elif growth_rate < -20:
            return f"Urgent: Address declining {column} - investigate root causes"
        elif growth_rate < -5:
            return f"Investigate factors causing decline in {column}"
        else:
            return f"Maintain current strategies for {column}"

    def _analyze_missing_data_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns in missing data"""
        try:
            findings = []
            
            missing_summary = df.isnull().sum()
            high_missing_cols = missing_summary[missing_summary > len(df) * 0.1].index.tolist()
            
            for col in high_missing_cols:
                missing_pct = (missing_summary[col] / len(df)) * 100
                findings.append({
                    'type': 'data_quality',
                    'metric': col,
                    'value': missing_pct,
                    'description': f'{col} has {missing_pct:.1f}% missing values',
                    'impact': 'high' if missing_pct > 50 else 'medium',
                    'recommendation': f'Review data collection process for {col}'
                })
            
            return findings
            
        except Exception as e:
            logger.error(f"Error analyzing missing data patterns: {str(e)}")
            return []

    def _analyze_performance_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze performance distribution for a column"""
        try:
            if not pd.api.types.is_numeric_dtype(df[column]):
                return None
                
            col_data = df[column].dropna()
            if len(col_data) == 0:
                return None
                
            q25, q75 = col_data.quantile([0.25, 0.75])
            top_performers = col_data[col_data >= q75]
            bottom_performers = col_data[col_data <= q25]
            
            return {
                'description': f'{column} shows performance variation with top 25% averaging {top_performers.mean():.2f}',
                'actions': [
                    f'Analyze top performers in {column} to identify success factors',
                    f'Develop improvement plans for bottom performers in {column}',
                    f'Consider benchmarking strategies for {column}'
                ],
                'impact': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance distribution: {str(e)}")
            return None

    def _identify_growth_opportunities(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify growth opportunities"""
        try:
            opportunities = []
            
            # Time-based growth analysis
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                for num_col in numerical_cols:
                    growth_rate = self._calculate_growth_rate(df, num_col)
                    if growth_rate > 5:  # Positive growth
                        opportunities.append({
                            'category': 'growth',
                            'priority': 'high' if growth_rate > 15 else 'medium',
                            'title': f'Scale {num_col} Growth',
                            'description': f'{num_col} showing {growth_rate:.1f}% growth - opportunity to scale',
                            'actions': [
                                f'Analyze factors driving {num_col} growth',
                                f'Develop scaling strategy for {num_col}',
                                f'Monitor {num_col} growth sustainability'
                            ],
                            'expected_impact': 'high',
                            'timeline': '6-12 months'
                        })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying growth opportunities: {str(e)}")
            return []

    def _generate_risk_mitigation_recommendations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk mitigation recommendations"""
        try:
            recommendations = []
            
            # Data quality risks
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 10:
                recommendations.append({
                    'category': 'data_quality',
                    'priority': 'high',
                    'title': 'Improve Data Quality',
                    'description': f'High missing data ({missing_percentage:.1f}%) poses analytical risks',
                    'actions': [
                        'Implement data validation processes',
                        'Review data collection procedures',
                        'Establish data quality monitoring'
                    ],
                    'expected_impact': 'high',
                    'timeline': '1-3 months'
                })
            
            # Outlier risks
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                outliers = self._detect_outliers(df[col])
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    recommendations.append({
                        'category': 'outlier_management',
                        'priority': 'medium',
                        'title': f'Manage {col} Outliers',
                        'description': f'High outlier count in {col} may indicate process issues',
                        'actions': [
                            f'Investigate root causes of {col} outliers',
                            f'Implement {col} monitoring thresholds',
                            f'Review {col} data collection process'
                        ],
                        'expected_impact': 'medium',
                        'timeline': '2-4 months'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk mitigation recommendations: {str(e)}")
            return []

    def _identify_process_improvements(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify process improvement opportunities"""
        try:
            improvements = []
            
            # Efficiency improvements based on correlations
            if 'correlation_analysis' in analysis_results:
                strong_correlations = self._identify_strong_correlations(analysis_results['correlation_analysis'])
                for corr in strong_correlations:
                    improvements.append({
                        'category': 'process_efficiency',
                        'priority': 'medium',
                        'title': f'Optimize {corr["var1"]}-{corr["var2"]} Relationship',
                        'description': f'Strong correlation between {corr["var1"]} and {corr["var2"]} suggests process optimization opportunity',
                        'actions': [
                            f'Analyze the relationship between {corr["var1"]} and {corr["var2"]}',
                            f'Develop integrated approach for {corr["var1"]} and {corr["var2"]}',
                            f'Monitor combined performance metrics'
                        ],
                        'expected_impact': 'medium',
                        'timeline': '3-6 months'
                    })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error identifying process improvements: {str(e)}")
            return []

    def generate_insights(self, df: pd.DataFrame, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate basic insights (simplified version for compatibility)"""
        try:
            # Create a basic analysis results structure
            analysis_results = {
                'basic_statistics': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'missing_values': df.isnull().sum().sum()
                },
                'correlation_analysis': {
                    'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {}
                }
            }
            
            return self.generate_business_insights(df, analysis_results, user_prompt)
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {'error': str(e)}

    def generate_comprehensive_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights in list format"""
        try:
            insights_dict = self.generate_business_insights(df, analysis_results)
            
            # Convert to list format
            insights_list = []
            
            for category, content in insights_dict.items():
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            item['category'] = category
                            insights_list.append(item)
                elif isinstance(content, dict):
                    content['category'] = category
                    insights_list.append(content)
            
            return insights_list
            
        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {str(e)}")
            return []

    def _assess_data_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data timeliness"""
        try:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return {'assessment': 'Cannot assess - no datetime columns'}
            
            # Check the most recent date
            latest_date = df[datetime_cols[0]].max()
            current_date = pd.Timestamp.now()
            
            if pd.isna(latest_date):
                return {'assessment': 'Cannot assess - no valid dates'}
            
            days_old = (current_date - latest_date).days
            
            assessment = 'Current' if days_old <= 1 else \
                        'Recent' if days_old <= 7 else \
                        'Moderate' if days_old <= 30 else \
                        'Outdated'
            
            return {
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'assessment': assessment
            }
            
        except Exception as e:
            logger.error(f"Error assessing data timeliness: {str(e)}")
            return {'assessment': 'Unknown'}

    def _analyze_growth_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze growth patterns in the data"""
        try:
            growth_patterns = []
            
            # Check for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numerical_cols) == 0:
                return growth_patterns
            
            # Sort by the first datetime column
            df_sorted = df.sort_values(datetime_cols[0])
            
            for col in numerical_cols:
                try:
                    # Calculate growth rate
                    growth_rate = self._calculate_growth_rate(df_sorted, col)
                    
                    # Determine pattern type
                    if growth_rate > 10:
                        pattern_type = 'strong_growth'
                        description = f'{col} shows strong growth of {growth_rate:.1f}%'
                    elif growth_rate > 2:
                        pattern_type = 'moderate_growth'
                        description = f'{col} shows moderate growth of {growth_rate:.1f}%'
                    elif growth_rate < -10:
                        pattern_type = 'strong_decline'
                        description = f'{col} shows strong decline of {abs(growth_rate):.1f}%'
                    elif growth_rate < -2:
                        pattern_type = 'moderate_decline'
                        description = f'{col} shows moderate decline of {abs(growth_rate):.1f}%'
                    else:
                        pattern_type = 'stable'
                        description = f'{col} remains relatively stable'
                    
                    growth_patterns.append({
                        'metric': col,
                        'pattern_type': pattern_type,
                        'growth_rate': growth_rate,
                        'description': description,
                        'business_impact': self._assess_growth_impact(growth_rate, col)
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing growth for {col}: {str(e)}")
                    continue
            
            return growth_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing growth patterns: {str(e)}")
            return []

    def _assess_growth_impact(self, growth_rate: float, metric: str) -> str:
        """Assess the business impact of growth rate"""
        try:
            if growth_rate > 20:
                return f"High positive impact - {metric} growth creates significant opportunities"
            elif growth_rate > 10:
                return f"Moderate positive impact - {metric} growth is beneficial"
            elif growth_rate > 0:
                return f"Low positive impact - {metric} shows slight improvement"
            elif growth_rate > -10:
                return f"Low negative impact - {metric} shows slight decline"
            elif growth_rate > -20:
                return f"Moderate negative impact - {metric} decline needs attention"
            else:
                return f"High negative impact - {metric} decline requires immediate action"
        except Exception as e:
            return f"Unable to assess impact for {metric}"

    def _analyze_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze trends in the data"""
        try:
            trends = []
            
            # Check for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) == 0 or len(numerical_cols) == 0:
                return trends
            
            # Sort by the first datetime column
            df_sorted = df.sort_values(datetime_cols[0])
            
            for col in numerical_cols:
                try:
                    # Calculate trend direction
                    trend_direction = self._calculate_trend_direction(df_sorted, col)
                    
                    # Calculate trend strength
                    if len(df_sorted) >= 2:
                        x = np.arange(len(df_sorted))
                        y = df_sorted[col].values
                        
                        # Remove NaN values
                        mask = ~np.isnan(y)
                        if np.sum(mask) >= 2:
                            x_clean = x[mask]
                            y_clean = y[mask]
                            
                            slope, _, r_value, p_value, _ = stats.linregress(x_clean, y_clean)
                            
                            trends.append({
                                'metric': col,
                                'direction': trend_direction,
                                'strength': abs(r_value),
                                'slope': slope,
                                'significance': p_value,
                                'description': f'{col} shows {trend_direction} trend with {abs(r_value):.2f} correlation strength'
                            })
                    
                except Exception as e:
                    logger.error(f"Error analyzing trend for {col}: {str(e)}")
                    continue
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return []

    def _analyze_performance(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze performance metrics"""
        try:
            performance_analysis = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Calculate performance metrics
                    mean_val = col_data.mean()
                    median_val = col_data.median()
                    std_val = col_data.std()
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    # Performance categorization
                    q25, q75 = col_data.quantile([0.25, 0.75])
                    
                    performance_analysis.append({
                        'metric': col,
                        'mean': mean_val,
                        'median': median_val,
                        'std': std_val,
                        'coefficient_of_variation': cv,
                        'performance_range': {
                            'low': q25,
                            'high': q75
                        },
                        'consistency': 'high' if cv < 0.1 else 'medium' if cv < 0.3 else 'low',
                        'description': f'{col} performance analysis: mean={mean_val:.2f}, consistency={cv:.2f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing performance for {col}: {str(e)}")
                    continue
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return []

    def _analyze_customer_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze customer-related patterns"""
        try:
            patterns = []
            
            # Look for customer-related columns
            customer_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['customer', 'client', 'user', 'account'])]
            
            for col in customer_cols:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Numerical customer analysis
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            patterns.append({
                                'type': 'customer_metric',
                                'column': col,
                                'average': col_data.mean(),
                                'distribution': 'normal' if self._test_normality(col_data) else 'skewed',
                                'outliers': len(self._detect_outliers(col_data)),
                                'description': f'Customer metric {col} analysis'
                            })
                    else:
                        # Categorical customer analysis
                        value_counts = df[col].value_counts()
                        if len(value_counts) > 0:
                            patterns.append({
                                'type': 'customer_segment',
                                'column': col,
                                'segments': len(value_counts),
                                'top_segment': value_counts.index[0],
                                'concentration': value_counts.iloc[0] / len(df),
                                'description': f'Customer segmentation in {col}'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing customer pattern for {col}: {str(e)}")
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing customer patterns: {str(e)}")
            return []

    def _analyze_financial_metrics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze financial metrics"""
        try:
            financial_analysis = []
            
            # Look for financial columns
            financial_keywords = ['revenue', 'sales', 'cost', 'profit', 'price', 'amount', 'value', 'income']
            financial_cols = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in financial_keywords)]
            
            for col in financial_cols:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            total = col_data.sum()
                            average = col_data.mean()
                            growth_rate = self._calculate_growth_rate(df, col) if self._is_time_series_column(df, col) else 0
                            
                            financial_analysis.append({
                                'metric': col,
                                'total': total,
                                'average': average,
                                'growth_rate': growth_rate,
                                'volatility': col_data.std() / average if average != 0 else 0,
                                'trend': 'positive' if growth_rate > 0 else 'negative' if growth_rate < 0 else 'stable',
                                'description': f'Financial metric {col}: total={total:.2f}, growth={growth_rate:.1f}%'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing financial metric for {col}: {str(e)}")
                    continue
            
            return financial_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {str(e)}")
            return []

    def _analyze_operational_metrics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze operational metrics"""
        try:
            operational_analysis = []
            
            # Look for operational columns
            operational_keywords = ['quantity', 'volume', 'count', 'rate', 'efficiency', 'productivity']
            operational_cols = [col for col in df.columns if any(keyword in col.lower() 
                              for keyword in operational_keywords)]
            
            for col in operational_cols:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            operational_analysis.append({
                                'metric': col,
                                'average': col_data.mean(),
                                'efficiency_score': min(100, (col_data.mean() / col_data.max()) * 100) if col_data.max() > 0 else 0,
                                'consistency': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0,
                                'outliers': len(self._detect_outliers(col_data)),
                                'description': f'Operational metric {col} analysis'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing operational metric for {col}: {str(e)}")
                    continue
            
            return operational_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing operational metrics: {str(e)}")
            return []

    def _analyze_market_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze market patterns"""
        try:
            market_patterns = []
            
            # Look for market-related columns
            market_keywords = ['market', 'segment', 'region', 'category', 'product', 'service']
            market_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in market_keywords)]
            
            for col in market_cols:
                try:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # Categorical market analysis
                        value_counts = df[col].value_counts()
                        if len(value_counts) > 0:
                            market_patterns.append({
                                'type': 'market_segment',
                                'column': col,
                                'segments': len(value_counts),
                                'dominant_segment': value_counts.index[0],
                                'market_share': value_counts.iloc[0] / len(df),
                                'diversity': len(value_counts) / len(df),
                                'description': f'Market segmentation in {col}'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing market pattern for {col}: {str(e)}")
                    continue
            
            return market_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing market patterns: {str(e)}")
            return []

    def _analyze_risk_factors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze risk factors"""
        try:
            risk_factors = []
            
            # Data quality risks
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 10:
                risk_factors.append({
                    'type': 'data_quality',
                    'risk_level': 'high' if missing_percentage > 30 else 'medium',
                    'description': f'High missing data: {missing_percentage:.1f}%',
                    'impact': 'Affects analysis reliability',
                    'mitigation': 'Improve data collection processes'
                })
            
            # Outlier risks
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                try:
                    outliers = self._detect_outliers(df[col])
                    outlier_percentage = len(outliers) / len(df) * 100
                    
                    if outlier_percentage > 5:
                        risk_factors.append({
                            'type': 'outlier_risk',
                            'column': col,
                            'risk_level': 'high' if outlier_percentage > 15 else 'medium',
                            'description': f'High outlier count in {col}: {outlier_percentage:.1f}%',
                            'impact': 'May skew analysis results',
                            'mitigation': f'Investigate and validate outliers in {col}'
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing risk for {col}: {str(e)}")
                    continue
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {str(e)}")
            return []

    def _analyze_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze opportunities"""
        try:
            opportunities = []
            
            # Growth opportunities
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                try:
                    if self._is_time_series_column(df, col):
                        growth_rate = self._calculate_growth_rate(df, col)
                        if growth_rate > 5:
                            opportunities.append({
                                'type': 'growth_opportunity',
                                'column': col,
                                'growth_rate': growth_rate,
                                'potential': 'high' if growth_rate > 20 else 'medium',
                                'description': f'{col} showing {growth_rate:.1f}% growth',
                                'action': f'Scale and optimize {col} growth strategies'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing opportunity for {col}: {str(e)}")
                    continue
            
            # Efficiency opportunities
            for col in numerical_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                        if cv > 0.3:  # High variability
                            opportunities.append({
                                'type': 'efficiency_opportunity',
                                'column': col,
                                'variability': cv,
                                'potential': 'medium',
                                'description': f'{col} shows high variability - optimization opportunity',
                                'action': f'Standardize and optimize {col} processes'
                            })
                            
                except Exception as e:
                    logger.error(f"Error analyzing efficiency opportunity for {col}: {str(e)}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing opportunities: {str(e)}")
            return []

    def _analyze_competitive_position(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze competitive position"""
        try:
            competitive_analysis = []
            
            # Performance benchmarking
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Calculate percentile rankings
                        percentiles = [25, 50, 75, 90]
                        percentile_values = [col_data.quantile(p/100) for p in percentiles]
                        
                        competitive_analysis.append({
                            'metric': col,
                            'benchmarks': {
                                'bottom_quartile': percentile_values[0],
                                'median': percentile_values[1],
                                'top_quartile': percentile_values[2],
                                'top_decile': percentile_values[3]
                            },
                            'competitive_position': self._assess_competitive_position(col_data),
                            'description': f'Competitive benchmarking for {col}'
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing competitive position for {col}: {str(e)}")
                    continue
            
            return competitive_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing competitive position: {str(e)}")
            return []

    def _assess_competitive_position(self, series: pd.Series) -> str:
        """Assess competitive position based on data distribution"""
        try:
            mean_val = series.mean()
            median_val = series.median()
            q75 = series.quantile(0.75)
            
            if mean_val >= q75:
                return 'strong'
            elif mean_val >= median_val:
                return 'competitive'
            else:
                return 'needs_improvement'
                
        except Exception as e:
            return 'unknown' 