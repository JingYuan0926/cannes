"""
Business Intelligence Engine Module

This module provides AI-powered business intelligence capabilities:
- OpenAI integration for intelligent insights
- Automated report generation
- Strategic recommendations
- Performance analysis
- Trend identification and forecasting
- Business metrics calculation

Integrates with DataAnalyzer and VisualizationEngine for comprehensive analytics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import openai
import os
from dotenv import load_dotenv
from .data_analyzer import DataAnalyzer
from .visualization_engine import VisualizationEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessIntelligenceEngine:
    """
    AI-powered business intelligence engine for comprehensive analytics
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the business intelligence engine
        
        Args:
            openai_api_key (str, optional): OpenAI API key for AI insights
        """
        self.data_analyzer = DataAnalyzer()
        self.visualization_engine = VisualizationEngine()
        
        # Initialize OpenAI client if API key is provided
        self.openai_client = None
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                openai.api_key = api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        else:
            logger.info("OpenAI API key not provided, using fallback analysis")
    
    def generate_comprehensive_report(self, df: pd.DataFrame, user_prompt: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive business intelligence report
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            user_prompt (str, optional): User's specific requirements or questions
            
        Returns:
            dict: Comprehensive business intelligence report
        """
        try:
            logger.info("Starting comprehensive business intelligence analysis")
            
            # Perform comprehensive data analysis
            analysis_results = self.data_analyzer.comprehensive_analysis(df)
            
            # Generate AI-powered insights
            ai_insights = self.generate_ai_insights(df, analysis_results, user_prompt)
            
            # Create visualizations
            visualizations = self.create_recommended_visualizations(df, analysis_results)
            
            # Generate strategic recommendations
            strategic_recommendations = self.generate_strategic_recommendations(df, analysis_results)
            
            # Create executive summary
            executive_summary = self.create_executive_summary(df, analysis_results, ai_insights)
            
            # Compile comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'dataset_overview': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict()
                },
                'executive_summary': executive_summary,
                'detailed_analysis': analysis_results,
                'ai_insights': ai_insights,
                'visualizations': visualizations,
                'strategic_recommendations': strategic_recommendations,
                'key_findings': self.extract_key_findings(analysis_results, ai_insights),
                'next_steps': self.generate_next_steps(analysis_results, strategic_recommendations)
            }
            
            logger.info("Comprehensive business intelligence report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}
    
    def generate_ai_insights(self, df: pd.DataFrame, analysis_results: Dict, user_prompt: str = None) -> Dict[str, Any]:
        """
        Generate AI-powered insights using OpenAI
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Results from data analysis
            user_prompt (str, optional): User's specific questions
            
        Returns:
            dict: AI-generated insights
        """
        try:
            if not self.openai_client:
                return self.generate_fallback_insights(df, analysis_results, user_prompt)
            
            # Prepare context for OpenAI
            context = self.prepare_analysis_context(df, analysis_results, user_prompt)
            
            # Generate insights using OpenAI
            insights = {}
            
            # General insights
            insights['general_insights'] = self.get_openai_insights(
                context, 
                "Provide general business insights and observations about this dataset"
            )
            
            # Trend analysis insights
            insights['trend_insights'] = self.get_openai_insights(
                context,
                "Analyze trends and patterns in the data and their business implications"
            )
            
            # Performance insights
            insights['performance_insights'] = self.get_openai_insights(
                context,
                "Evaluate performance metrics and identify areas for improvement"
            )
            
            # Predictive insights
            insights['predictive_insights'] = self.get_openai_insights(
                context,
                "Provide predictive insights and forecast potential future scenarios"
            )
            
            # Risk assessment
            insights['risk_assessment'] = self.get_openai_insights(
                context,
                "Identify potential risks and challenges based on the data patterns"
            )
            
            # Opportunity identification
            insights['opportunities'] = self.get_openai_insights(
                context,
                "Identify business opportunities and growth potential from the data"
            )
            
            # User-specific insights
            if user_prompt:
                insights['user_specific_insights'] = self.get_openai_insights(
                    context,
                    f"Answer this specific question about the data: {user_prompt}"
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return self.generate_fallback_insights(df, analysis_results, user_prompt)
    
    def get_openai_insights(self, context: str, prompt: str) -> Dict[str, Any]:
        """
        Get insights from OpenAI for a specific prompt
        
        Args:
            context (str): Data context
            prompt (str): Specific prompt for insights
            
        Returns:
            dict: OpenAI insights response
        """
        try:
            full_prompt = f"""
            Context: {context}
            
            Task: {prompt}
            
            Please provide insights in JSON format with the following structure:
            {{
                "insights": ["insight1", "insight2", "insight3"],
                "key_findings": ["finding1", "finding2"],
                "recommendations": ["recommendation1", "recommendation2"],
                "confidence_level": "high/medium/low"
            }}
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business intelligence expert analyzing data to provide actionable insights."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to text if needed
            try:
                parsed_response = json.loads(response_text)
                return parsed_response
            except json.JSONDecodeError:
                return {
                    "insights": [response_text],
                    "key_findings": ["AI analysis completed"],
                    "recommendations": ["Review the insights provided"],
                    "confidence_level": "medium"
                }
                
        except Exception as e:
            logger.error(f"Error getting OpenAI insights: {str(e)}")
            return {
                "insights": ["Unable to generate AI insights"],
                "key_findings": ["Analysis completed with limitations"],
                "recommendations": ["Manual review recommended"],
                "confidence_level": "low"
            }
    
    def generate_fallback_insights(self, df: pd.DataFrame, analysis_results: Dict, user_prompt: str = None) -> Dict[str, Any]:
        """
        Generate insights without OpenAI (fallback method)
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            user_prompt (str, optional): User prompt
            
        Returns:
            dict: Fallback insights
        """
        try:
            insights = {
                'general_insights': self.generate_statistical_insights(df, analysis_results),
                'trend_insights': self.generate_trend_insights(df, analysis_results),
                'performance_insights': self.generate_performance_insights(df, analysis_results),
                'data_quality_insights': self.generate_data_quality_insights(df, analysis_results),
                'correlation_insights': self.generate_correlation_insights(df, analysis_results)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating fallback insights: {str(e)}")
            return {}
    
    def prepare_analysis_context(self, df: pd.DataFrame, analysis_results: Dict, user_prompt: str = None) -> str:
        """
        Prepare context for OpenAI analysis
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            user_prompt (str, optional): User prompt
            
        Returns:
            str: Formatted context for OpenAI
        """
        try:
            context_parts = []
            
            # Dataset overview
            context_parts.append(f"Dataset Overview:")
            context_parts.append(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            context_parts.append(f"- Columns: {', '.join(df.columns)}")
            
            # Data types
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numerical_cols:
                context_parts.append(f"- Numerical columns: {', '.join(numerical_cols)}")
            if categorical_cols:
                context_parts.append(f"- Categorical columns: {', '.join(categorical_cols)}")
            
            # Key statistics
            if 'basic_statistics' in analysis_results:
                context_parts.append("\nKey Statistics:")
                stats = analysis_results['basic_statistics']
                
                if 'numerical_stats' in stats:
                    for col, col_stats in stats['numerical_stats'].items():
                        if isinstance(col_stats, dict) and 'mean' in col_stats:
                            context_parts.append(f"- {col}: mean={col_stats['mean']:.2f}, std={col_stats.get('std', 0):.2f}")
            
            # Data quality issues
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                if 'missing_percentage' in quality:
                    missing_cols = [col for col, pct in quality['missing_percentage'].items() if pct > 0]
                    if missing_cols:
                        context_parts.append(f"\nData Quality Issues:")
                        context_parts.append(f"- Columns with missing values: {', '.join(missing_cols)}")
            
            # Correlations
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                if 'correlation_analysis' in desc and 'strong_correlations' in desc['correlation_analysis']:
                    strong_corrs = desc['correlation_analysis']['strong_correlations']
                    if strong_corrs:
                        context_parts.append("\nStrong Correlations:")
                        for corr in strong_corrs[:3]:  # Top 3 correlations
                            context_parts.append(f"- {corr['variable1']} & {corr['variable2']}: {corr['correlation']:.3f}")
            
            # User prompt
            if user_prompt:
                context_parts.append(f"\nUser Question: {user_prompt}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing analysis context: {str(e)}")
            return "Dataset analysis context unavailable"
    
    def create_recommended_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
        """
        Create recommended visualizations based on analysis
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            
        Returns:
            list: List of recommended visualizations
        """
        try:
            visualizations = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Distribution plots for numerical columns
            for col in numerical_cols[:3]:  # Limit to first 3 columns
                try:
                    viz_result = self.visualization_engine.create_visualization(
                        df, 
                        {
                            'chart_type': 'histogram',
                            'x_column': col,
                            'title': f'Distribution of {col}'
                        }
                    )
                    if viz_result:
                        visualizations.append({
                            'type': 'distribution',
                            'column': col,
                            'chart': viz_result
                        })
                except Exception as e:
                    logger.error(f"Error creating distribution plot for {col}: {str(e)}")
            
            # Correlation heatmap if multiple numerical columns
            if len(numerical_cols) > 1:
                try:
                    viz_result = self.visualization_engine.create_visualization(
                        df,
                        {
                            'chart_type': 'heatmap',
                            'columns': numerical_cols,
                            'title': 'Correlation Matrix'
                        }
                    )
                    if viz_result:
                        visualizations.append({
                            'type': 'correlation',
                            'columns': numerical_cols,
                            'chart': viz_result
                        })
                except Exception as e:
                    logger.error(f"Error creating correlation heatmap: {str(e)}")
            
            # Category distribution for categorical columns
            for col in categorical_cols[:2]:  # Limit to first 2 columns
                try:
                    viz_result = self.visualization_engine.create_visualization(
                        df,
                        {
                            'chart_type': 'bar',
                            'x_column': col,
                            'aggregation': 'count',
                            'title': f'Distribution of {col}'
                        }
                    )
                    if viz_result:
                        visualizations.append({
                            'type': 'category_distribution',
                            'column': col,
                            'chart': viz_result
                        })
                except Exception as e:
                    logger.error(f"Error creating category distribution for {col}: {str(e)}")
            
            # Scatter plots for strong correlations
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                if 'correlation_analysis' in desc and 'strong_correlations' in desc['correlation_analysis']:
                    strong_corrs = desc['correlation_analysis']['strong_correlations']
                    for corr in strong_corrs[:2]:  # Top 2 correlations
                        try:
                            viz_result = self.visualization_engine.create_visualization(
                                df,
                                {
                                    'chart_type': 'scatter',
                                    'x_column': corr['variable1'],
                                    'y_column': corr['variable2'],
                                    'title': f'{corr["variable1"]} vs {corr["variable2"]}'
                                }
                            )
                            if viz_result:
                                visualizations.append({
                                    'type': 'correlation_scatter',
                                    'variables': [corr['variable1'], corr['variable2']],
                                    'correlation': corr['correlation'],
                                    'chart': viz_result
                                })
                        except Exception as e:
                            logger.error(f"Error creating scatter plot: {str(e)}")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating recommended visualizations: {str(e)}")
            return []
    
    def generate_strategic_recommendations(self, df: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
        """
        Generate strategic business recommendations
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            
        Returns:
            list: Strategic recommendations
        """
        try:
            recommendations = []
            
            # Data quality recommendations
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                
                # Missing values recommendations
                if 'missing_percentage' in quality:
                    high_missing = {col: pct for col, pct in quality['missing_percentage'].items() if pct > 10}
                    if high_missing:
                        recommendations.append({
                            'category': 'Data Quality',
                            'priority': 'High',
                            'title': 'Address Missing Data',
                            'description': f"Columns with high missing values: {', '.join(high_missing.keys())}",
                            'action_items': [
                                'Investigate root causes of missing data',
                                'Implement data collection improvements',
                                'Consider imputation strategies',
                                'Establish data validation processes'
                            ],
                            'expected_impact': 'Improved analysis accuracy and reliability'
                        })
                
                # Duplicate data recommendations
                if quality.get('duplicate_percentage', 0) > 5:
                    recommendations.append({
                        'category': 'Data Quality',
                        'priority': 'Medium',
                        'title': 'Remove Duplicate Records',
                        'description': f"Found {quality['duplicate_percentage']:.1f}% duplicate records",
                        'action_items': [
                            'Identify and remove duplicate records',
                            'Implement deduplication processes',
                            'Review data entry procedures'
                        ],
                        'expected_impact': 'Cleaner dataset and more accurate analysis'
                    })
            
            # Performance optimization recommendations
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                
                # Outlier management
                if 'correlation_analysis' in desc and 'strong_correlations' in desc['correlation_analysis']:
                    strong_corrs = desc['correlation_analysis']['strong_correlations']
                    if strong_corrs:
                        recommendations.append({
                            'category': 'Performance Optimization',
                            'priority': 'Medium',
                            'title': 'Leverage Strong Correlations',
                            'description': f"Found {len(strong_corrs)} strong correlations between variables",
                            'action_items': [
                                'Investigate causal relationships',
                                'Optimize processes based on correlations',
                                'Implement predictive models',
                                'Monitor correlation stability over time'
                            ],
                            'expected_impact': 'Better predictive capabilities and process optimization'
                        })
            
            # Business process recommendations
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if not df[col].empty:
                    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    
                    if cv > 0.5:  # High variability
                        recommendations.append({
                            'category': 'Process Improvement',
                            'priority': 'Medium',
                            'title': f'Reduce Variability in {col}',
                            'description': f"High variability detected (CV: {cv:.2f})",
                            'action_items': [
                                'Analyze root causes of variability',
                                'Implement process standardization',
                                'Establish quality control measures',
                                'Monitor performance metrics regularly'
                            ],
                            'expected_impact': 'More consistent performance and predictable outcomes'
                        })
            
            # Growth opportunity recommendations
            if 'predictive_analysis' in analysis_results:
                pred = analysis_results['predictive_analysis']
                
                if 'trend_analysis' in pred:
                    trends = pred['trend_analysis']
                    positive_trends = [col for col, trend in trends.items() 
                                     if isinstance(trend, dict) and trend.get('trend_direction') == 'Increasing']
                    
                    if positive_trends:
                        recommendations.append({
                            'category': 'Growth Opportunity',
                            'priority': 'High',
                            'title': 'Capitalize on Positive Trends',
                            'description': f"Positive trends identified in: {', '.join(positive_trends)}",
                            'action_items': [
                                'Invest in areas showing positive trends',
                                'Accelerate growth initiatives',
                                'Allocate resources strategically',
                                'Monitor trend sustainability'
                            ],
                            'expected_impact': 'Enhanced growth potential and competitive advantage'
                        })
            
            # Technology recommendations
            if df.shape[0] > 10000:  # Large dataset
                recommendations.append({
                    'category': 'Technology',
                    'priority': 'Low',
                    'title': 'Consider Big Data Solutions',
                    'description': f"Dataset size: {df.shape[0]} records",
                    'action_items': [
                        'Evaluate big data processing tools',
                        'Implement data warehousing solutions',
                        'Consider cloud-based analytics platforms',
                        'Optimize data storage and retrieval'
                    ],
                    'expected_impact': 'Improved processing speed and scalability'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {str(e)}")
            return []
    
    def create_executive_summary(self, df: pd.DataFrame, analysis_results: Dict, ai_insights: Dict) -> Dict[str, Any]:
        """
        Create executive summary of the analysis
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            ai_insights (dict): AI insights
            
        Returns:
            dict: Executive summary
        """
        try:
            summary = {
                'overview': {
                    'dataset_size': f"{df.shape[0]:,} records with {df.shape[1]} variables",
                    'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                    'analysis_type': 'Comprehensive Business Intelligence Analysis'
                },
                'key_metrics': {},
                'critical_findings': [],
                'priority_actions': [],
                'business_impact': {}
            }
            
            # Key metrics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                summary['key_metrics'] = {
                    'total_numerical_variables': len(numerical_cols),
                    'data_completeness': f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%",
                    'data_quality_score': self.calculate_data_quality_score(df, analysis_results)
                }
            
            # Critical findings
            critical_findings = []
            
            # Data quality findings
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                if quality.get('duplicate_percentage', 0) > 10:
                    critical_findings.append(f"High duplicate rate: {quality['duplicate_percentage']:.1f}%")
                
                missing_issues = [col for col, pct in quality.get('missing_percentage', {}).items() if pct > 20]
                if missing_issues:
                    critical_findings.append(f"Significant missing data in: {', '.join(missing_issues)}")
            
            # Correlation findings
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                if 'correlation_analysis' in desc:
                    strong_corrs = desc['correlation_analysis'].get('strong_correlations', [])
                    if strong_corrs:
                        critical_findings.append(f"Strong correlations identified between {len(strong_corrs)} variable pairs")
            
            # AI insights findings
            if ai_insights and 'general_insights' in ai_insights:
                general = ai_insights['general_insights']
                if isinstance(general, dict) and 'key_findings' in general:
                    critical_findings.extend(general['key_findings'][:2])  # Top 2 AI findings
            
            summary['critical_findings'] = critical_findings
            
            # Priority actions
            priority_actions = []
            
            # From data quality
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                if quality.get('duplicate_percentage', 0) > 5:
                    priority_actions.append("Implement data deduplication process")
                
                high_missing = [col for col, pct in quality.get('missing_percentage', {}).items() if pct > 15]
                if high_missing:
                    priority_actions.append("Address missing data in critical columns")
            
            # From AI insights
            if ai_insights and 'general_insights' in ai_insights:
                general = ai_insights['general_insights']
                if isinstance(general, dict) and 'recommendations' in general:
                    priority_actions.extend(general['recommendations'][:2])  # Top 2 AI recommendations
            
            summary['priority_actions'] = priority_actions
            
            # Business impact
            summary['business_impact'] = {
                'data_reliability': 'High' if summary['key_metrics'].get('data_quality_score', 0) > 0.8 else 'Medium',
                'analytical_readiness': 'Ready' if len(critical_findings) < 3 else 'Needs Improvement',
                'insights_confidence': 'High' if self.openai_client else 'Medium'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating executive summary: {str(e)}")
            return {}
    
    def extract_key_findings(self, analysis_results: Dict, ai_insights: Dict) -> List[str]:
        """
        Extract key findings from analysis results
        
        Args:
            analysis_results (dict): Analysis results
            ai_insights (dict): AI insights
            
        Returns:
            list: Key findings
        """
        try:
            findings = []
            
            # Data quality findings
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                
                # Missing data findings
                missing_cols = [col for col, pct in quality.get('missing_percentage', {}).items() if pct > 0]
                if missing_cols:
                    findings.append(f"Missing data detected in {len(missing_cols)} columns")
                
                # Duplicate findings
                if quality.get('duplicate_percentage', 0) > 0:
                    findings.append(f"Duplicate records: {quality['duplicate_percentage']:.1f}%")
            
            # Statistical findings
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                
                # Correlation findings
                if 'correlation_analysis' in desc:
                    strong_corrs = desc['correlation_analysis'].get('strong_correlations', [])
                    if strong_corrs:
                        findings.append(f"Strong correlations found between {len(strong_corrs)} variable pairs")
                
                # Distribution findings
                if 'distribution_analysis' in desc:
                    dist_analysis = desc['distribution_analysis']
                    non_normal = [col for col, dist in dist_analysis.items() 
                                 if isinstance(dist, dict) and dist.get('normality_test', {}).get('is_normal') == False]
                    if non_normal:
                        findings.append(f"Non-normal distributions detected in {len(non_normal)} variables")
            
            # Trend findings
            if 'predictive_analysis' in analysis_results:
                pred = analysis_results['predictive_analysis']
                
                if 'trend_analysis' in pred:
                    trends = pred['trend_analysis']
                    increasing_trends = [col for col, trend in trends.items() 
                                       if isinstance(trend, dict) and trend.get('trend_direction') == 'Increasing']
                    decreasing_trends = [col for col, trend in trends.items() 
                                       if isinstance(trend, dict) and trend.get('trend_direction') == 'Decreasing']
                    
                    if increasing_trends:
                        findings.append(f"Positive trends identified in {len(increasing_trends)} variables")
                    if decreasing_trends:
                        findings.append(f"Negative trends identified in {len(decreasing_trends)} variables")
            
            # AI insights findings
            for insight_type, insights in ai_insights.items():
                if isinstance(insights, dict) and 'key_findings' in insights:
                    findings.extend(insights['key_findings'][:2])  # Top 2 findings per category
            
            return findings[:10]  # Return top 10 findings
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {str(e)}")
            return []
    
    def generate_next_steps(self, analysis_results: Dict, strategic_recommendations: List[Dict]) -> List[Dict]:
        """
        Generate next steps based on analysis
        
        Args:
            analysis_results (dict): Analysis results
            strategic_recommendations (list): Strategic recommendations
            
        Returns:
            list: Next steps with priorities and timelines
        """
        try:
            next_steps = []
            
            # Immediate actions (1-2 weeks)
            immediate_actions = []
            
            # Data quality issues
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                if quality.get('duplicate_percentage', 0) > 5:
                    immediate_actions.append({
                        'action': 'Remove duplicate records',
                        'timeline': '1-2 weeks',
                        'responsibility': 'Data Team',
                        'resources_needed': 'Data cleaning tools'
                    })
            
            # Short-term actions (1-3 months)
            short_term_actions = []
            
            # From strategic recommendations
            high_priority_recs = [rec for rec in strategic_recommendations if rec.get('priority') == 'High']
            for rec in high_priority_recs[:3]:  # Top 3 high-priority recommendations
                short_term_actions.append({
                    'action': rec['title'],
                    'timeline': '1-3 months',
                    'responsibility': 'Business Team',
                    'resources_needed': 'Budget allocation and team assignment'
                })
            
            # Medium-term actions (3-6 months)
            medium_term_actions = []
            
            medium_priority_recs = [rec for rec in strategic_recommendations if rec.get('priority') == 'Medium']
            for rec in medium_priority_recs[:2]:  # Top 2 medium-priority recommendations
                medium_term_actions.append({
                    'action': rec['title'],
                    'timeline': '3-6 months',
                    'responsibility': 'Operations Team',
                    'resources_needed': 'Process improvement initiatives'
                })
            
            # Long-term actions (6+ months)
            long_term_actions = []
            
            low_priority_recs = [rec for rec in strategic_recommendations if rec.get('priority') == 'Low']
            for rec in low_priority_recs[:2]:  # Top 2 low-priority recommendations
                long_term_actions.append({
                    'action': rec['title'],
                    'timeline': '6+ months',
                    'responsibility': 'Strategic Planning Team',
                    'resources_needed': 'Long-term investment planning'
                })
            
            # Compile next steps
            if immediate_actions:
                next_steps.append({
                    'category': 'Immediate Actions',
                    'timeline': '1-2 weeks',
                    'actions': immediate_actions
                })
            
            if short_term_actions:
                next_steps.append({
                    'category': 'Short-term Actions',
                    'timeline': '1-3 months',
                    'actions': short_term_actions
                })
            
            if medium_term_actions:
                next_steps.append({
                    'category': 'Medium-term Actions',
                    'timeline': '3-6 months',
                    'actions': medium_term_actions
                })
            
            if long_term_actions:
                next_steps.append({
                    'category': 'Long-term Actions',
                    'timeline': '6+ months',
                    'actions': long_term_actions
                })
            
            return next_steps
            
        except Exception as e:
            logger.error(f"Error generating next steps: {str(e)}")
            return []
    
    def calculate_data_quality_score(self, df: pd.DataFrame, analysis_results: Dict) -> float:
        """
        Calculate overall data quality score
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Analysis results
            
        Returns:
            float: Data quality score (0-1)
        """
        try:
            score_components = []
            
            # Completeness score
            completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            score_components.append(completeness)
            
            # Uniqueness score
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                uniqueness = 1 - (quality.get('duplicate_percentage', 0) / 100)
                score_components.append(uniqueness)
            
            # Consistency score (based on data type issues)
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                type_issues = len(quality.get('data_types_issues', []))
                consistency = max(0, 1 - (type_issues / df.shape[1]))
                score_components.append(consistency)
            
            # Overall score
            overall_score = sum(score_components) / len(score_components) if score_components else 0
            
            return round(overall_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {str(e)}")
            return 0.0
    
    def generate_statistical_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Generate statistical insights without AI"""
        try:
            insights = []
            key_findings = []
            recommendations = []
            
            # Basic statistics insights
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                insights.append(f"Dataset contains {len(numerical_cols)} numerical variables")
                
                # High variability detection
                high_var_cols = []
                for col in numerical_cols:
                    if not df[col].empty and df[col].std() / df[col].mean() > 0.5:
                        high_var_cols.append(col)
                
                if high_var_cols:
                    key_findings.append(f"High variability detected in: {', '.join(high_var_cols)}")
                    recommendations.append("Investigate causes of high variability")
            
            return {
                'insights': insights,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'confidence_level': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error generating statistical insights: {str(e)}")
            return {}
    
    def generate_trend_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Generate trend insights without AI"""
        try:
            insights = []
            key_findings = []
            recommendations = []
            
            if 'predictive_analysis' in analysis_results:
                pred = analysis_results['predictive_analysis']
                
                if 'trend_analysis' in pred:
                    trends = pred['trend_analysis']
                    
                    increasing = [col for col, trend in trends.items() 
                                if isinstance(trend, dict) and trend.get('trend_direction') == 'Increasing']
                    decreasing = [col for col, trend in trends.items() 
                                if isinstance(trend, dict) and trend.get('trend_direction') == 'Decreasing']
                    
                    if increasing:
                        insights.append(f"Positive trends in: {', '.join(increasing)}")
                        key_findings.append("Growth opportunities identified")
                        recommendations.append("Capitalize on positive trends")
                    
                    if decreasing:
                        insights.append(f"Negative trends in: {', '.join(decreasing)}")
                        key_findings.append("Areas requiring attention identified")
                        recommendations.append("Address declining trends")
            
            return {
                'insights': insights,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'confidence_level': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {str(e)}")
            return {}
    
    def generate_performance_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Generate performance insights without AI"""
        try:
            insights = []
            key_findings = []
            recommendations = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                # Performance variability analysis
                performance_scores = {}
                for col in numerical_cols:
                    if not df[col].empty:
                        cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                        performance_scores[col] = 1 - min(cv, 1)  # Higher score for lower variability
                
                best_performer = max(performance_scores, key=performance_scores.get)
                worst_performer = min(performance_scores, key=performance_scores.get)
                
                insights.append(f"Most consistent performance: {best_performer}")
                insights.append(f"Highest variability: {worst_performer}")
                
                key_findings.append("Performance consistency varies across metrics")
                recommendations.append("Focus on improving consistency in variable metrics")
            
            return {
                'insights': insights,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'confidence_level': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {str(e)}")
            return {}
    
    def generate_data_quality_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Generate data quality insights without AI"""
        try:
            insights = []
            key_findings = []
            recommendations = []
            
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                
                # Missing data insights
                missing_pct = quality.get('missing_percentage', {})
                high_missing = [col for col, pct in missing_pct.items() if pct > 10]
                
                if high_missing:
                    insights.append(f"High missing data in: {', '.join(high_missing)}")
                    key_findings.append("Data completeness issues identified")
                    recommendations.append("Implement data collection improvements")
                
                # Duplicate data insights
                dup_pct = quality.get('duplicate_percentage', 0)
                if dup_pct > 5:
                    insights.append(f"Duplicate records: {dup_pct:.1f}%")
                    key_findings.append("Data duplication detected")
                    recommendations.append("Implement deduplication process")
            
            return {
                'insights': insights,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'confidence_level': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error generating data quality insights: {str(e)}")
            return {}
    
    def generate_correlation_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Generate correlation insights without AI"""
        try:
            insights = []
            key_findings = []
            recommendations = []
            
            if 'descriptive_analysis' in analysis_results:
                desc = analysis_results['descriptive_analysis']
                
                if 'correlation_analysis' in desc:
                    corr_analysis = desc['correlation_analysis']
                    strong_corrs = corr_analysis.get('strong_correlations', [])
                    
                    if strong_corrs:
                        insights.append(f"Strong correlations found: {len(strong_corrs)} pairs")
                        
                        # Identify strongest correlation
                        strongest = max(strong_corrs, key=lambda x: abs(x['correlation']))
                        insights.append(f"Strongest correlation: {strongest['variable1']} & {strongest['variable2']} ({strongest['correlation']:.3f})")
                        
                        key_findings.append("Significant variable relationships identified")
                        recommendations.append("Investigate causal relationships")
                        recommendations.append("Consider predictive modeling opportunities")
            
            return {
                'insights': insights,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'confidence_level': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error generating correlation insights: {str(e)}")
            return {}
    
    def generate_insights(self, df: pd.DataFrame, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate insights for the given dataset
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            user_prompt (str, optional): User's specific question or focus area
            
        Returns:
            dict: Generated insights
        """
        try:
            # Perform comprehensive analysis
            analysis_results = self.data_analyzer.comprehensive_analysis(df)
            
            # Generate AI insights
            ai_insights = self.generate_ai_insights(df, analysis_results, user_prompt)
            
            # Combine insights
            insights = {
                'timestamp': datetime.now().isoformat(),
                'dataset_summary': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict()
                },
                'ai_insights': ai_insights,
                'key_findings': self.extract_key_findings(analysis_results, ai_insights),
                'recommendations': self.generate_strategic_recommendations(df, analysis_results)[:5],  # Top 5 recommendations
                'data_quality_score': self.calculate_data_quality_score(df, analysis_results)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {'error': str(e)} 