"""
Configuration file for AI-Powered Data Analysis System

This file contains all configuration settings for the analysis system.
"""

import os
from typing import Dict, List, Any

class Config:
    """Configuration settings for the analysis system"""
    
    # Flask settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 3040))
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
    
    # Analysis settings
    RESULTS_FOLDER = 'results'
    MAX_ANALYSIS_RESULTS = 100  # Maximum number of analysis results to keep
    
    # Analytics configuration
    ANALYTICS_CONFIG = {
        'descriptive': {
            'algorithms': ['kmeans_clustering', 'dbscan_clustering', 'pca_analysis', 'tsne_analysis'],
            'default_algorithm': 'kmeans_clustering',
            'description': 'Clustering and dimensionality reduction to understand data patterns'
        },
        'predictive': {
            'algorithms': ['linear_regression', 'ridge_regression', 'decision_tree_regression', 
                          'logistic_regression', 'random_forest_classification', 'svm_classification'],
            'default_algorithm': 'linear_regression',
            'description': 'Regression and classification to predict future outcomes'
        },
        'prescriptive': {
            'algorithms': ['linear_programming', 'genetic_algorithm', 'collaborative_filtering', 
                          'content_based_filtering', 'portfolio_optimization', 'resource_allocation'],
            'default_algorithm': 'linear_programming',
            'description': 'Optimization and recommendation algorithms for decision making'
        },
        'diagnostic': {
            'algorithms': ['decision_tree_analysis', 'feature_importance_analysis', 'causal_inference', 
                          'correlation_analysis', 'anomaly_detection', 'root_cause_analysis'],
            'default_algorithm': 'decision_tree_analysis',
            'description': 'Feature importance and causal analysis to understand why things happened'
        }
    }
    
    # AI prompts for analysis strategy
    AI_PROMPTS = {
        'dataset_analysis': """
        Analyze the following dataset characteristics and recommend the best analytics approach:
        
        Dataset Info:
        - Shape: {shape}
        - Columns: {columns}
        - Data Types: {dtypes}
        - Missing Values: {missing_values}
        - Statistical Summary: {stats_summary}
        
        User Goal: {goal}
        
        Based on this information, recommend:
        1. The most appropriate analytics category (descriptive, predictive, prescriptive, or diagnostic)
        2. The specific algorithm to use
        3. The reasoning behind your recommendation
        4. Expected insights and outcomes
        
        Respond in JSON format with the following structure:
        {{
            "recommended_category": "category_name",
            "recommended_algorithm": "algorithm_name",
            "reasoning": "explanation of why this approach is best",
            "expected_insights": ["insight1", "insight2", "insight3"],
            "alternative_approaches": [
                {{"category": "category", "algorithm": "algorithm", "reason": "reason"}}
            ]
        }}
        """,
        
        'insights_generation': """
        Based on the analysis results, generate comprehensive insights, justifications, conclusions, and recommendations:
        
        Analysis Results:
        {analysis_results}
        
        User Goal: {goal}
        
        Generate:
        1. Key insights from the analysis
        2. Justifications for the findings
        3. Conclusions drawn from the data
        4. Actionable recommendations
        
        Respond in JSON format with the following structure:
        {{
            "insights": ["insight1", "insight2", "insight3"],
            "justifications": ["justification1", "justification2"],
            "conclusions": ["conclusion1", "conclusion2"],
            "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
        }}
        """
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': 'analysis.log',
                'mode': 'a',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    @classmethod
    def get_analytics_algorithms(cls, category: str) -> List[str]:
        """Get available algorithms for a specific analytics category"""
        return cls.ANALYTICS_CONFIG.get(category, {}).get('algorithms', [])
    
    @classmethod
    def get_default_algorithm(cls, category: str) -> str:
        """Get default algorithm for a specific analytics category"""
        return cls.ANALYTICS_CONFIG.get(category, {}).get('default_algorithm', '')
    
    @classmethod
    def get_category_description(cls, category: str) -> str:
        """Get description for a specific analytics category"""
        return cls.ANALYTICS_CONFIG.get(category, {}).get('description', '')
    
    @classmethod
    def validate_algorithm(cls, category: str, algorithm: str) -> bool:
        """Validate if an algorithm is available for a category"""
        return algorithm in cls.get_analytics_algorithms(category)
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all available analytics categories"""
        return list(cls.ANALYTICS_CONFIG.keys())
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        return {
            'flask_config': {
                'host': cls.HOST,
                'port': cls.PORT,
                'debug': cls.DEBUG,
                'env': cls.FLASK_ENV
            },
            'openai_config': {
                'model': cls.OPENAI_MODEL,
                'temperature': cls.OPENAI_TEMPERATURE,
                'api_key_set': bool(cls.OPENAI_API_KEY)
            },
            'analytics_categories': cls.get_all_categories(),
            'max_file_size': f"{cls.MAX_CONTENT_LENGTH / (1024*1024):.0f}MB",
            'allowed_extensions': list(cls.ALLOWED_EXTENSIONS)
        } 