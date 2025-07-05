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
import openai
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

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
            
            if openai.api_key:
                response = openai.ChatCompletion.create(
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
            else:
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
            for analysis in results.get('analyses', []):
                all_insights.extend(analysis.get('insights', []))
            
            # Create summary for AI
            summary = {
                'goal': goal,
                'total_analyses': len(results.get('analyses', [])),
                'total_graphs': len(results.get('graphs', [])),
                'analytics_types': list(results.get('analytics_summary', {}).keys()),
                'key_insights': all_insights[:10]  # Top 10 insights
            }
            
            if openai.api_key:
                system_prompt = """You are an expert data scientist providing executive summary insights.
                Based on the analysis results, provide a comprehensive summary with:
                1. Justification for the analysis approach
                2. Key conclusions from the findings
                3. Actionable recommendations
                
                Return JSON with this structure:
                {
                    "justification": "Why this analysis approach was chosen and executed",
                    "conclusions": ["Key finding 1", "Key finding 2", "Key finding 3"],
                    "recommendations": ["Action 1", "Action 2", "Action 3"]
                }
                """
                
                user_message = f"""
                Analysis Summary:
                - Goal: {summary['goal']}
                - Total Analyses: {summary['total_analyses']}
                - Analytics Types: {summary['analytics_types']}
                - Key Insights: {summary['key_insights']}
                
                Please provide executive summary insights.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=800,
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
        """Generate fallback insights when OpenAI is not available"""
        analytics_types = list(results.get('analytics_summary', {}).keys())
        total_analyses = len(results.get('analyses', []))
        
        justification = f"Performed {total_analyses} analyses across {len(analytics_types)} analytics types to address the goal: {goal}. The analysis approach included {', '.join(analytics_types)} to provide comprehensive insights."
        
        conclusions = [
            f"Successfully completed {total_analyses} different analytical approaches",
            f"Generated {len(results.get('graphs', []))} visualizations to support findings",
            "Identified key patterns and relationships in the data"
        ]
        
        recommendations = [
            "Review the detailed analysis results for specific insights",
            "Consider implementing the suggested optimizations",
            "Monitor key metrics identified in the analysis"
        ]
        
        return {
            'justification': justification,
            'conclusions': conclusions,
            'recommendations': recommendations
        } 