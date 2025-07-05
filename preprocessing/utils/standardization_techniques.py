"""
Standardization Techniques Utility

This module provides comprehensive information about all available preprocessing techniques,
their parameters, use cases, and recommendations for different data types and scenarios.
"""

class StandardizationTechniques:
    """Comprehensive guide to all available preprocessing techniques"""
    
    def __init__(self):
        self.techniques = self._initialize_techniques()
    
    def _initialize_techniques(self):
        """Initialize all available preprocessing techniques with their details"""
        return {
            'standardize_numerical_features': {
                'description': 'Standardize numerical features using various scaling methods',
                'methods': {
                    'z_score': {
                        'description': 'Standard normalization (mean=0, std=1)',
                        'use_case': 'When features have different scales and normal distribution',
                        'parameters': {},
                        'pros': ['Preserves relationships', 'Works well with normal distributions'],
                        'cons': ['Sensitive to outliers', 'Assumes normal distribution']
                    },
                    'min_max': {
                        'description': 'Scale features to a fixed range [0,1]',
                        'use_case': 'When you need bounded features or non-negative values',
                        'parameters': {'feature_range': '(0,1) default, can be customized'},
                        'pros': ['Bounded output', 'Preserves zero values'],
                        'cons': ['Very sensitive to outliers', 'Can compress data']
                    },
                    'robust': {
                        'description': 'Scale using median and IQR, robust to outliers',
                        'use_case': 'When data contains outliers',
                        'parameters': {'quantile_range': '(25.0, 75.0) default'},
                        'pros': ['Robust to outliers', 'Good for skewed data'],
                        'cons': ['May not center data at zero', 'Less interpretable']
                    },
                    'quantile_uniform': {
                        'description': 'Transform to uniform distribution',
                        'use_case': 'When you want uniform distribution regardless of original shape',
                        'parameters': {'n_quantiles': '1000 default'},
                        'pros': ['Creates uniform distribution', 'Reduces outlier impact'],
                        'cons': ['Loses original distribution info', 'Non-linear transformation']
                    },
                    'quantile_normal': {
                        'description': 'Transform to normal distribution',
                        'use_case': 'When algorithms assume normal distribution',
                        'parameters': {'n_quantiles': '1000 default'},
                        'pros': ['Creates normal distribution', 'Good for ML algorithms'],
                        'cons': ['Loses original distribution info', 'Non-linear transformation']
                    }
                },
                'parameters': {
                    'columns': 'List of columns, "all", "numerical", or specific column names',
                    'method': 'Scaling method to use'
                }
            },
            
            'normalize_categorical_features': {
                'description': 'Encode categorical features using various methods',
                'methods': {
                    'label_encoding': {
                        'description': 'Assign integer labels to categories',
                        'use_case': 'Ordinal data or when memory is limited',
                        'parameters': {},
                        'pros': ['Memory efficient', 'Simple', 'Works with ordinal data'],
                        'cons': ['Implies ordering', 'Not suitable for nominal data']
                    },
                    'one_hot_encoding': {
                        'description': 'Create binary columns for each category',
                        'use_case': 'Nominal categorical data with few categories',
                        'parameters': {'dummy_na': 'True to handle missing values'},
                        'pros': ['No false ordering', 'Works with all ML algorithms'],
                        'cons': ['High dimensionality', 'Sparse matrices', 'Memory intensive']
                    },
                    'frequency_encoding': {
                        'description': 'Replace categories with their frequency counts',
                        'use_case': 'When category frequency is informative',
                        'parameters': {},
                        'pros': ['Single column output', 'Captures frequency info'],
                        'cons': ['Different categories may have same frequency', 'Loses category identity']
                    },
                    'binary_encoding': {
                        'description': 'Encode using binary representation',
                        'use_case': 'High cardinality categorical data',
                        'parameters': {},
                        'pros': ['More compact than one-hot', 'Handles high cardinality'],
                        'cons': ['Less interpretable', 'Creates artificial relationships']
                    },
                    'target_encoding': {
                        'description': 'Replace categories with target variable statistics',
                        'use_case': 'When category-target relationship is strong',
                        'parameters': {'target_column': 'Required target variable'},
                        'pros': ['Captures target relationship', 'Single column output'],
                        'cons': ['Risk of overfitting', 'Requires target variable']
                    }
                },
                'parameters': {
                    'columns': 'List of columns, "all", "categorical", or specific column names',
                    'method': 'Encoding method to use'
                }
            },
            
            'handle_text_features': {
                'description': 'Process text data using NLP techniques',
                'methods': {
                    'tfidf': {
                        'description': 'Term Frequency-Inverse Document Frequency vectorization',
                        'use_case': 'Text classification, document similarity',
                        'parameters': {'max_features': 'Maximum number of features to extract'},
                        'pros': ['Captures term importance', 'Good for text classification'],
                        'cons': ['High dimensionality', 'Loses word order']
                    },
                    'count_vectorizer': {
                        'description': 'Count-based text vectorization',
                        'use_case': 'Simple text analysis, when term frequency matters',
                        'parameters': {'max_features': 'Maximum number of features to extract'},
                        'pros': ['Simple and interpretable', 'Fast computation'],
                        'cons': ['Ignores term importance', 'High dimensionality']
                    },
                    'text_preprocessing': {
                        'description': 'Basic text cleaning (lowercase, remove punctuation)',
                        'use_case': 'Preparing text for further processing',
                        'parameters': {},
                        'pros': ['Standardizes text', 'Reduces noise'],
                        'cons': ['May lose important information', 'Basic cleaning only']
                    }
                },
                'parameters': {
                    'columns': 'List of text columns to process',
                    'method': 'Text processing method',
                    'max_features': 'Maximum number of features for vectorization'
                }
            },
            
            'engineer_datetime_features': {
                'description': 'Extract meaningful features from datetime columns',
                'methods': {
                    'standard_extraction': {
                        'description': 'Extract standard datetime components',
                        'use_case': 'When temporal patterns are important',
                        'parameters': {'extract': 'List of components to extract'},
                        'pros': ['Captures temporal patterns', 'Interpretable features'],
                        'cons': ['Increases dimensionality', 'May lose some temporal info']
                    }
                },
                'extractable_features': {
                    'year': 'Year component',
                    'month': 'Month component (1-12)',
                    'day': 'Day of month (1-31)',
                    'hour': 'Hour component (0-23)',
                    'weekday': 'Day of week (0-6)',
                    'quarter': 'Quarter of year (1-4)',
                    'is_weekend': 'Boolean indicating weekend',
                    'days_since_epoch': 'Days since Unix epoch'
                },
                'parameters': {
                    'columns': 'List of datetime columns to process',
                    'extract': 'List of features to extract'
                }
            },
            
            'create_polynomial_features': {
                'description': 'Create polynomial and interaction features',
                'methods': {
                    'polynomial': {
                        'description': 'Generate polynomial features up to specified degree',
                        'use_case': 'Capturing non-linear relationships',
                        'parameters': {
                            'degree': 'Polynomial degree (2 or 3 recommended)',
                            'interaction_only': 'Only interaction terms, no powers',
                            'include_bias': 'Include bias (constant) term'
                        },
                        'pros': ['Captures non-linear relationships', 'No domain knowledge needed'],
                        'cons': ['Exponential feature growth', 'Overfitting risk', 'Interpretability loss']
                    }
                },
                'parameters': {
                    'columns': 'Numerical columns for polynomial features',
                    'degree': 'Polynomial degree',
                    'interaction_only': 'Boolean for interaction-only features',
                    'include_bias': 'Boolean for including bias term'
                }
            },
            
            'apply_feature_selection': {
                'description': 'Select most relevant features',
                'methods': {
                    'variance_threshold': {
                        'description': 'Remove features with low variance',
                        'use_case': 'Removing quasi-constant features',
                        'parameters': {'threshold': 'Variance threshold (0.01 default)'},
                        'pros': ['Removes uninformative features', 'Fast computation'],
                        'cons': ['May remove important low-variance features', 'Univariate only']
                    },
                    'correlation_threshold': {
                        'description': 'Remove highly correlated features',
                        'use_case': 'Reducing multicollinearity',
                        'parameters': {'threshold': 'Correlation threshold (0.95 default)'},
                        'pros': ['Reduces multicollinearity', 'Improves model stability'],
                        'cons': ['May remove important correlated features', 'Pairwise only']
                    },
                    'mutual_info': {
                        'description': 'Select features based on mutual information',
                        'use_case': 'Capturing non-linear relationships with target',
                        'parameters': {'k': 'Number of features to select'},
                        'pros': ['Captures non-linear relationships', 'Model-agnostic'],
                        'cons': ['Requires target variable', 'Computationally expensive']
                    }
                },
                'parameters': {
                    'columns': 'Columns to apply feature selection',
                    'method': 'Feature selection method',
                    'k': 'Number of features to select (for some methods)'
                }
            },
            
            'detect_and_handle_outliers': {
                'description': 'Detect and remove outliers using various methods',
                'methods': {
                    'isolation_forest': {
                        'description': 'Isolation Forest algorithm for outlier detection',
                        'use_case': 'High-dimensional data with complex outlier patterns',
                        'parameters': {'contamination': 'Expected proportion of outliers'},
                        'pros': ['Works in high dimensions', 'No assumptions about data distribution'],
                        'cons': ['Parameter tuning needed', 'May remove valid edge cases']
                    },
                    'local_outlier_factor': {
                        'description': 'Local Outlier Factor for density-based detection',
                        'use_case': 'When outliers are in low-density regions',
                        'parameters': {'contamination': 'Expected proportion of outliers'},
                        'pros': ['Considers local density', 'Good for varying densities'],
                        'cons': ['Sensitive to parameters', 'Computationally expensive']
                    },
                    'one_class_svm': {
                        'description': 'One-Class SVM for novelty detection',
                        'use_case': 'When you have clean training data',
                        'parameters': {'nu': 'Upper bound on training errors'},
                        'pros': ['Robust method', 'Works with non-linear boundaries'],
                        'cons': ['Requires parameter tuning', 'Computationally expensive']
                    },
                    'elliptic_envelope': {
                        'description': 'Assumes data comes from known distribution',
                        'use_case': 'When data approximately follows multivariate normal',
                        'parameters': {'contamination': 'Expected proportion of outliers'},
                        'pros': ['Fast computation', 'Good for Gaussian data'],
                        'cons': ['Assumes elliptical distribution', 'Poor for non-Gaussian data']
                    }
                },
                'parameters': {
                    'columns': 'Numerical columns to check for outliers',
                    'method': 'Outlier detection method'
                }
            },
            
            'apply_dimensionality_reduction': {
                'description': 'Reduce feature dimensionality',
                'methods': {
                    'pca': {
                        'description': 'Principal Component Analysis',
                        'use_case': 'Linear dimensionality reduction, visualization',
                        'parameters': {'n_components': 'Number of components to keep'},
                        'pros': ['Removes correlation', 'Interpretable components', 'Fast'],
                        'cons': ['Linear only', 'Loses interpretability', 'Sensitive to scaling']
                    },
                    'factor_analysis': {
                        'description': 'Factor Analysis for latent factors',
                        'use_case': 'When looking for underlying factors',
                        'parameters': {'n_components': 'Number of factors'},
                        'pros': ['Finds latent factors', 'Handles noise well'],
                        'cons': ['Assumes linear relationships', 'Requires factor interpretation']
                    },
                    'tsne': {
                        'description': 't-SNE for non-linear dimensionality reduction',
                        'use_case': 'Visualization of high-dimensional data',
                        'parameters': {'n_components': 'Output dimensions (2 or 3 typical)'},
                        'pros': ['Preserves local structure', 'Great for visualization'],
                        'cons': ['Computationally expensive', 'Not deterministic', 'Only for visualization']
                    }
                },
                'parameters': {
                    'columns': 'Numerical columns for dimensionality reduction',
                    'method': 'Dimensionality reduction method',
                    'n_components': 'Number of output dimensions'
                }
            },
            
            'create_interaction_features': {
                'description': 'Create interaction features between variables',
                'methods': {
                    'multiplicative': {
                        'description': 'Multiply pairs of features',
                        'use_case': 'When features have multiplicative relationships',
                        'parameters': {},
                        'pros': ['Captures multiplicative effects', 'Simple to interpret'],
                        'cons': ['Exponential feature growth', 'May create noise']
                    },
                    'additive': {
                        'description': 'Add pairs of features',
                        'use_case': 'When features have additive relationships',
                        'parameters': {},
                        'pros': ['Simple combination', 'Interpretable'],
                        'cons': ['May not capture complex interactions', 'Feature redundancy']
                    },
                    'ratio_features': {
                        'description': 'Create ratios between features',
                        'use_case': 'When relative relationships matter',
                        'parameters': {},
                        'pros': ['Captures relative importance', 'Scale-invariant'],
                        'cons': ['Division by zero issues', 'May amplify noise']
                    }
                },
                'parameters': {
                    'columns': 'Numerical columns for interaction features',
                    'method': 'Type of interaction to create'
                }
            },
            
            'handle_imbalanced_data': {
                'description': 'Handle class imbalance in datasets',
                'methods': {
                    'smote': {
                        'description': 'Synthetic Minority Oversampling Technique',
                        'use_case': 'When minority class is underrepresented',
                        'parameters': {'target_column': 'Target variable column'},
                        'pros': ['Creates synthetic samples', 'Preserves class distribution'],
                        'cons': ['May create noise', 'Requires numerical features']
                    },
                    'adasyn': {
                        'description': 'Adaptive Synthetic Sampling',
                        'use_case': 'When different regions need different sampling',
                        'parameters': {'target_column': 'Target variable column'},
                        'pros': ['Adaptive sampling', 'Focuses on difficult regions'],
                        'cons': ['More complex than SMOTE', 'Parameter sensitive']
                    },
                    'random_oversample': {
                        'description': 'Random oversampling of minority class',
                        'use_case': 'Simple oversampling approach',
                        'parameters': {'target_column': 'Target variable column'},
                        'pros': ['Simple and fast', 'Preserves original data'],
                        'cons': ['May cause overfitting', 'Duplicates exact samples']
                    },
                    'random_undersample': {
                        'description': 'Random undersampling of majority class',
                        'use_case': 'When dataset is very large',
                        'parameters': {'target_column': 'Target variable column'},
                        'pros': ['Reduces dataset size', 'Fast training'],
                        'cons': ['Loses information', 'May remove important samples']
                    }
                },
                'parameters': {
                    'target_column': 'Target variable for balancing',
                    'method': 'Balancing method to use'
                }
            }
        }
    
    def get_all_techniques(self):
        """Get all available techniques with their descriptions"""
        return {name: {
            'description': details['description'],
            'methods': list(details['methods'].keys()) if 'methods' in details else [],
            'parameters': details['parameters']
        } for name, details in self.techniques.items()}
    
    def get_technique_details(self, technique_name):
        """Get detailed information about a specific technique"""
        if technique_name in self.techniques:
            return self.techniques[technique_name]
        else:
            return None
    
    def get_method_details(self, technique_name, method_name):
        """Get detailed information about a specific method within a technique"""
        if technique_name in self.techniques and 'methods' in self.techniques[technique_name]:
            methods = self.techniques[technique_name]['methods']
            if method_name in methods:
                return methods[method_name]
        return None
    
    def recommend_techniques_for_data_type(self, data_type, use_case=None):
        """Recommend techniques based on data type and use case"""
        recommendations = []
        
        if data_type == 'numerical':
            recommendations.extend([
                'standardize_numerical_features',
                'detect_and_handle_outliers',
                'create_polynomial_features',
                'apply_dimensionality_reduction'
            ])
        
        elif data_type == 'categorical':
            recommendations.extend([
                'normalize_categorical_features'
            ])
        
        elif data_type == 'text':
            recommendations.extend([
                'handle_text_features'
            ])
        
        elif data_type == 'datetime':
            recommendations.extend([
                'engineer_datetime_features'
            ])
        
        elif data_type == 'mixed':
            recommendations.extend([
                'standardize_numerical_features',
                'normalize_categorical_features',
                'handle_text_features',
                'engineer_datetime_features',
                'apply_feature_selection'
            ])
        
        # Add use case specific recommendations
        if use_case == 'machine_learning':
            recommendations.extend([
                'apply_feature_selection',
                'handle_imbalanced_data'
            ])
        
        elif use_case == 'deep_learning':
            recommendations.extend([
                'standardize_numerical_features',
                'normalize_categorical_features'
            ])
        
        elif use_case == 'visualization':
            recommendations.extend([
                'apply_dimensionality_reduction',
                'standardize_numerical_features'
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_preprocessing_pipeline_template(self, data_types, use_case='machine_learning'):
        """Get a template preprocessing pipeline based on data types and use case"""
        pipeline = []
        
        # Basic preprocessing based on data types
        if 'numerical' in data_types:
            pipeline.append({
                'technique': 'standardize_numerical_features',
                'parameters': {'method': 'robust'},
                'reason': 'Standardize numerical features for better model performance',
                'priority': 8
            })
        
        if 'categorical' in data_types:
            pipeline.append({
                'technique': 'normalize_categorical_features',
                'parameters': {'method': 'one_hot_encoding'},
                'reason': 'Encode categorical features for machine learning',
                'priority': 7
            })
        
        if 'text' in data_types:
            pipeline.append({
                'technique': 'handle_text_features',
                'parameters': {'method': 'tfidf', 'max_features': 100},
                'reason': 'Convert text to numerical features',
                'priority': 6
            })
        
        if 'datetime' in data_types:
            pipeline.append({
                'technique': 'engineer_datetime_features',
                'parameters': {'extract': ['year', 'month', 'day', 'weekday']},
                'reason': 'Extract temporal features from datetime columns',
                'priority': 5
            })
        
        # Use case specific additions
        if use_case == 'machine_learning':
            pipeline.extend([
                {
                    'technique': 'detect_and_handle_outliers',
                    'parameters': {'method': 'isolation_forest'},
                    'reason': 'Remove outliers that may affect model performance',
                    'priority': 4
                },
                {
                    'technique': 'apply_feature_selection',
                    'parameters': {'method': 'variance_threshold'},
                    'reason': 'Remove low-variance features',
                    'priority': 3
                }
            ])
        
        elif use_case == 'deep_learning':
            pipeline.extend([
                {
                    'technique': 'apply_dimensionality_reduction',
                    'parameters': {'method': 'pca', 'n_components': 50},
                    'reason': 'Reduce dimensionality for faster training',
                    'priority': 4
                }
            ])
        
        elif use_case == 'visualization':
            pipeline.extend([
                {
                    'technique': 'apply_dimensionality_reduction',
                    'parameters': {'method': 'tsne', 'n_components': 2},
                    'reason': 'Reduce to 2D for visualization',
                    'priority': 2
                }
            ])
        
        return sorted(pipeline, key=lambda x: x['priority'], reverse=True)
    
    def validate_technique_parameters(self, technique_name, parameters):
        """Validate parameters for a specific technique"""
        if technique_name not in self.techniques:
            return False, f"Unknown technique: {technique_name}"
        
        technique = self.techniques[technique_name]
        required_params = technique.get('parameters', {})
        
        # Basic validation (can be extended)
        for param, description in required_params.items():
            if param not in parameters and 'required' in description.lower():
                return False, f"Missing required parameter: {param}"
        
        return True, "Parameters are valid" 