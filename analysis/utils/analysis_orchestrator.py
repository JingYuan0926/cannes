"""
AI-Powered Analysis Orchestrator

This module coordinates different types of analytics based on user goals and dataset characteristics.
It uses OpenAI to determine the best analysis approach and orchestrates the execution of
descriptive, predictive, prescriptive, and diagnostic analytics.

Created: 2025-01-21
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    OPENAI_AVAILABLE = bool(os.getenv('OPENAI_API_KEY'))
except ImportError:
    openai_client = None
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available - using fallback strategies")

from .descriptive_analytics import DescriptiveAnalytics
from .predictive_analytics import PredictiveAnalytics
from .prescriptive_analytics import PrescriptiveAnalytics
from .diagnostic_analytics import DiagnosticAnalytics

logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """
    Main orchestrator that uses AI to determine the best analysis approach
    """
    
    def __init__(self):
        self.descriptive = DescriptiveAnalytics()
        self.predictive = PredictiveAnalytics()
        self.prescriptive = PrescriptiveAnalytics()
        self.diagnostic = DiagnosticAnalytics()
    
    def analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics for AI decision making"""
        try:
            characteristics = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'sample_size': len(df),
                'feature_count': len(df.columns)
            }
            
            # Statistical summary for numerical columns
            if characteristics['numerical_columns']:
                numerical_summary = df[characteristics['numerical_columns']].describe()
                characteristics['numerical_summary'] = numerical_summary.to_dict()
            
            # Categorical summary
            if characteristics['categorical_columns']:
                categorical_summary = {}
                for col in characteristics['categorical_columns'][:5]:  # Limit to first 5 categorical columns
                    categorical_summary[col] = {
                        'unique_count': df[col].nunique(),
                        'top_values': df[col].value_counts().head(3).to_dict()
                    }
                characteristics['categorical_summary'] = categorical_summary
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing dataset characteristics: {e}")
            return {}
    
    def get_ai_analysis_strategy(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Use OpenAI to determine the best analysis strategy"""
        try:
            characteristics = self.analyze_dataset_characteristics(df)
            
            system_prompt = """You are an expert data scientist. Based on the dataset characteristics and user goal, 
            recommend the most appropriate analytics approaches and specific algorithms to use.
            
            Available Analytics Types:
            1. DESCRIPTIVE (What happened?): Clustering (K-Means, DBSCAN), Dimensionality Reduction (PCA, t-SNE)
            2. PREDICTIVE (What will happen?): Regression (Linear, Ridge, Decision Tree), Classification (Logistic, Random Forest, SVM)
            3. PRESCRIPTIVE (What should we do?): Optimization algorithms, Recommendation systems
            4. DIAGNOSTIC (Why did it happen?): Feature importance analysis, Causal inference
            
            Return a JSON response with this exact structure:
            {
                "recommended_analytics": [
                    {
                        "type": "descriptive|predictive|prescriptive|diagnostic",
                        "algorithm": "specific_algorithm_name",
                        "priority": 1-5,
                        "justification": "Why this algorithm is suitable",
                        "expected_insights": "What insights this will provide"
                    }
                ],
                "analysis_sequence": ["type1", "type2", "type3"],
                "key_variables": ["important_column_names"],
                "success_metrics": ["metrics_to_evaluate_success"]
            }
            
            Recommend 4-6 analyses total, ensuring variety across different analytics types.
            """
            
            user_message = f"""
            Dataset Characteristics:
            - Shape: {characteristics.get('shape', 'Unknown')}
            - Columns: {characteristics.get('columns', [])}
            - Numerical columns: {characteristics.get('numerical_columns', [])}
            - Categorical columns: {characteristics.get('categorical_columns', [])}
            - Sample size: {characteristics.get('sample_size', 0)}
            - Missing values: {characteristics.get('missing_values', {})}
            
            User Goal: {goal}
            
            Please recommend the most appropriate analytics approaches and algorithms for this dataset and goal.
            """
            
            if OPENAI_AVAILABLE and openai_client:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=1500,
                        temperature=0.3
                    )
                    
                    strategy = json.loads(response.choices[0].message.content)
                    return strategy
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OpenAI response: {e}")
                    return self.get_fallback_strategy(df, goal)
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    return self.get_fallback_strategy(df, goal)
            else:
                logger.info("OpenAI not available, using fallback strategy")
                return self.get_fallback_strategy(df, goal)
                
        except Exception as e:
            logger.error(f"Error getting AI analysis strategy: {e}")
            return self.get_fallback_strategy(df, goal)
    
    def get_fallback_strategy(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Fallback strategy when OpenAI is not available"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        recommended_analytics = []
        
        # Descriptive Analytics
        if len(numerical_cols) >= 2:
            recommended_analytics.append({
                "type": "descriptive",
                "algorithm": "kmeans_clustering",
                "priority": 1,
                "justification": "Multiple numerical features suitable for clustering",
                "expected_insights": "Identify natural groupings in the data"
            })
        
        if len(numerical_cols) >= 3:
            recommended_analytics.append({
                "type": "descriptive",
                "algorithm": "pca_analysis",
                "priority": 2,
                "justification": "High dimensionality suitable for PCA",
                "expected_insights": "Understand main sources of variation"
            })
        
        # Predictive Analytics
        if len(numerical_cols) >= 2:
            recommended_analytics.append({
                "type": "predictive",
                "algorithm": "linear_regression",
                "priority": 1,
                "justification": "Multiple numerical features suitable for regression",
                "expected_insights": "Predict target variable based on features"
            })
        
        if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
            recommended_analytics.append({
                "type": "predictive",
                "algorithm": "random_forest_classification",
                "priority": 2,
                "justification": "Mixed data types suitable for classification",
                "expected_insights": "Classify data points into categories"
            })
        
        # Diagnostic Analytics
        if len(numerical_cols) >= 2:
            recommended_analytics.append({
                "type": "diagnostic",
                "algorithm": "feature_importance_analysis",
                "priority": 1,
                "justification": "Multiple features available for importance analysis",
                "expected_insights": "Understand which features drive the outcomes"
            })
        
        # Prescriptive Analytics
        if len(numerical_cols) >= 2:
            recommended_analytics.append({
                "type": "prescriptive",
                "algorithm": "linear_programming",
                "priority": 3,
                "justification": "Generate actionable recommendations based on optimization",
                "expected_insights": "Provide specific actions to achieve goals"
            })
        
        return {
            "recommended_analytics": recommended_analytics,
            "analysis_sequence": ["descriptive", "predictive", "diagnostic", "prescriptive"],
            "key_variables": numerical_cols[:3] + categorical_cols[:2],
            "success_metrics": ["accuracy", "explained_variance", "silhouette_score"]
        }
    
    def perform_comprehensive_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform comprehensive analysis based on AI recommendations"""
        try:
            analysis_id = str(uuid.uuid4())[:8]
            start_time = datetime.now()
            
            logger.info(f"Starting comprehensive analysis {analysis_id} for goal: {goal}")
            
            # Get AI-powered analysis strategy
            strategy = self.get_ai_analysis_strategy(df, goal)
            
            # Initialize results structure
            results = {
                'analysis_id': analysis_id,
                'goal': goal,
                'timestamp': start_time.isoformat(),
                'dataset_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                'strategy': strategy,
                'analyses': [],
                'graphs': [],
                'analytics_summary': {},
                'overall_insights': {
                    'justification': '',
                    'conclusions': [],
                    'recommendations': []
                }
            }
            
            # Execute analyses based on strategy
            for analysis_config in strategy.get('recommended_analytics', []):
                try:
                    analysis_type = analysis_config['type']
                    algorithm = analysis_config['algorithm']
                    
                    logger.info(f"Executing {analysis_type} analysis: {algorithm}")
                    
                    # Route to appropriate analytics engine
                    if analysis_type == 'descriptive':
                        analysis_result = self.descriptive.perform_analysis(df, algorithm, goal)
                    elif analysis_type == 'predictive':
                        analysis_result = self.predictive.perform_analysis(df, algorithm, goal)
                    elif analysis_type == 'prescriptive':
                        analysis_result = self.prescriptive.perform_analysis(df, algorithm, goal)
                    elif analysis_type == 'diagnostic':
                        analysis_result = self.diagnostic.perform_analysis(df, algorithm, goal)
                    else:
                        continue
                    
                    if analysis_result and 'error' not in analysis_result:
                        # Add analysis metadata
                        analysis_result['config'] = analysis_config
                        analysis_result['analysis_type'] = analysis_type
                        analysis_result['algorithm'] = algorithm
                        
                        results['analyses'].append(analysis_result)
                        
                        # Collect graphs
                        if 'graphs' in analysis_result:
                            results['graphs'].extend(analysis_result['graphs'])
                        
                        # Update analytics summary
                        if analysis_type not in results['analytics_summary']:
                            results['analytics_summary'][analysis_type] = []
                        results['analytics_summary'][analysis_type].append({
                            'algorithm': algorithm,
                            'status': 'completed',
                            'insights_count': len(analysis_result.get('insights', []))
                        })
                        
                        logger.info(f"Completed {analysis_type} analysis: {algorithm}")
                    else:
                        logger.warning(f"Failed {analysis_type} analysis: {algorithm}")
                        
                except Exception as e:
                    logger.error(f"Error in {analysis_type} analysis {algorithm}: {e}")
                    continue
            
            # Generate overall insights using AI
            results['overall_insights'] = self.generate_overall_insights(results, goal)
            
            # Calculate execution time
            end_time = datetime.now()
            results['execution_time'] = (end_time - start_time).total_seconds()
            results['completed_at'] = end_time.isoformat()
            
            logger.info(f"Comprehensive analysis {analysis_id} completed in {results['execution_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'analysis_id': analysis_id if 'analysis_id' in locals() else 'unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_overall_insights(self, results: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Generate overall insights, justification, conclusions, and recommendations"""
        try:
            # Collect all insights from individual analyses
            all_insights = []
            analysis_summaries = []
            
            for analysis in results.get('analyses', []):
                all_insights.extend(analysis.get('insights', []))
                analysis_summaries.append({
                    'algorithm': analysis.get('algorithm', 'unknown'),
                    'type': analysis.get('analysis_type', 'unknown'),
                    'insights_count': len(analysis.get('insights', [])),
                    'has_results': bool(analysis.get('results'))
                })
            
            # Create detailed summary for AI
            summary = {
                'goal': goal,
                'total_analyses': len(results.get('analyses', [])),
                'analysis_details': analysis_summaries,
                'key_insights': all_insights[:15],  # More insights for better context
                'graphs_generated': sum(len(analysis.get('graphs', [])) for analysis in results.get('analyses', []))
            }
            
            if OPENAI_AVAILABLE and openai_client:
                system_prompt = """You are a senior data scientist providing executive insights for business stakeholders.
                Based on the machine learning analysis results, provide specific, actionable insights that directly relate to the data and goal.
                
                IMPORTANT: 
                - Be specific about what the data reveals, not generic statements
                - Focus on actionable business insights, not technical details
                - Mention specific patterns, trends, or anomalies found
                - Provide concrete recommendations based on the findings
                
                Return JSON with this structure:
                {
                    "conclusions": [
                        "Specific finding about the data pattern/trend",
                        "Another concrete insight from the analysis", 
                        "Key business-relevant discovery"
                    ],
                    "recommendations": [
                        "Specific actionable recommendation",
                        "Another concrete action to take",
                        "Strategic next step based on findings"
                    ],
                    "key_patterns": [
                        "Important pattern discovered in the data",
                        "Significant correlation or relationship found"
                    ]
                }
                """
                
                user_message = f"""
                Analysis Goal: {summary['goal']}
                
                Analysis Results Summary:
                - Total ML algorithms applied: {summary['total_analyses']}
                - Visualizations generated: {summary['graphs_generated']}
                - Analysis types performed: {[a['algorithm'] for a in summary['analysis_details']]}
                
                Key Insights from Analysis:
                {chr(10).join('- ' + insight for insight in summary['key_insights'][:10])}
                
                Please provide specific, data-driven insights that a business stakeholder would find valuable.
                Focus on what the analysis actually discovered about the data, not generic statements about completing analyses.
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                insights = json.loads(response.choices[0].message.content)
                return insights
            else:
                return self.generate_fallback_insights(results, goal)
                
        except Exception as e:
            logger.error(f"Error generating overall insights: {e}")
            return self.generate_fallback_insights(results, goal)
    
    def generate_fallback_insights(self, results: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Generate meaningful fallback insights when OpenAI is not available"""
        analyses = results.get('analyses', [])
        total_graphs = sum(len(analysis.get('graphs', [])) for analysis in analyses)
        
        # Extract specific algorithms used
        algorithms_used = [analysis.get('algorithm', 'unknown') for analysis in analyses]
        unique_algorithms = list(set(algorithms_used))
        
        # Create more specific insights based on actual analysis results
        conclusions = []
        recommendations = []
        key_patterns = []
        
        # Analyze what types of analyses were successful
        if any('correlation' in alg.lower() for alg in algorithms_used):
            conclusions.append("Correlation analysis revealed relationships between key variables in your dataset")
            recommendations.append("Focus on the strongest correlations identified for predictive modeling")
        
        if any('cluster' in alg.lower() for alg in algorithms_used):
            conclusions.append("Clustering analysis identified distinct groups within your data")
            recommendations.append("Investigate the characteristics of each cluster for targeted strategies")
        
        if any('regression' in alg.lower() for alg in algorithms_used):
            conclusions.append("Regression analysis provided insights into predictive relationships")
            recommendations.append("Use the identified predictive factors for forecasting and decision making")
        
        if any('time' in alg.lower() or 'trend' in alg.lower() for alg in algorithms_used):
            conclusions.append("Time series analysis revealed temporal patterns in your data")
            recommendations.append("Monitor these trends for better timing of business decisions")
        
        # If no specific analysis types, provide general but meaningful insights
        if not conclusions:
            conclusions = [
                f"Applied {len(unique_algorithms)} different analytical approaches to understand your data patterns",
                f"Generated {total_graphs} visualizations highlighting key relationships and trends",
                "Identified statistical patterns that can inform data-driven decisions"
            ]
            
        if not recommendations:
            recommendations = [
                "Review the specific visualizations to understand data patterns",
                "Consider the identified relationships when making strategic decisions",
                "Use these insights as a foundation for further targeted analysis"
            ]
        
        key_patterns = [
            f"Applied advanced analytics including: {', '.join(unique_algorithms[:3])}",
            f"Generated comprehensive visualizations across {total_graphs} charts"
        ]
        
        return {
            'conclusions': conclusions,
            'recommendations': recommendations,
            'key_patterns': key_patterns
        } 