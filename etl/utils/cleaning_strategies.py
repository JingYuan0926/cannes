class CleaningStrategies:
    """Registry of all available data cleaning strategies and their configurations"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available cleaning strategies"""
        return {
            'handle_missing_values': {
                'description': 'Handle missing values in the dataset',
                'parameters': {
                    'columns': {
                        'type': 'list',
                        'description': 'List of columns to process (None for all)',
                        'default': None,
                        'optional': True
                    },
                    'strategy': {
                        'type': 'string',
                        'description': 'Strategy to handle missing values',
                        'options': ['drop', 'fill_mean', 'fill_median', 'fill_mode', 
                                  'forward_fill', 'backward_fill', 'interpolate', 
                                  'fill_value', 'fill_zero'],
                        'default': 'drop'
                    },
                    'fill_value': {
                        'type': 'any',
                        'description': 'Value to fill when strategy is fill_value',
                        'default': None,
                        'optional': True
                    }
                },
                'use_cases': [
                    'High missing data percentage',
                    'Systematic missing data patterns',
                    'Random missing values'
                ]
            },
            
            'remove_duplicates': {
                'description': 'Remove duplicate rows from the dataset',
                'parameters': {
                    'subset': {
                        'type': 'list',
                        'description': 'Columns to consider for duplicate detection',
                        'default': None,
                        'optional': True
                    },
                    'keep': {
                        'type': 'string',
                        'description': 'Which duplicate to keep',
                        'options': ['first', 'last', 'False'],
                        'default': 'first'
                    }
                },
                'use_cases': [
                    'Exact duplicate rows',
                    'Duplicate based on specific columns',
                    'Data entry errors'
                ]
            },
            
            'handle_outliers': {
                'description': 'Detect and handle outliers in numerical columns',
                'parameters': {
                    'columns': {
                        'type': 'list',
                        'description': 'Columns to process (None for all numerical)',
                        'default': None,
                        'optional': True
                    },
                    'method': {
                        'type': 'string',
                        'description': 'Method to detect outliers',
                        'options': ['iqr', 'z_score', 'isolation_forest', 'percentile', 'cap'],
                        'default': 'iqr'
                    },
                    'threshold': {
                        'type': 'float',
                        'description': 'Threshold for outlier detection',
                        'default': 1.5,
                        'range': [0.1, 5.0]
                    }
                },
                'use_cases': [
                    'Statistical outliers',
                    'Data entry errors',
                    'Measurement errors',
                    'Extreme values affecting analysis'
                ]
            },
            
            'standardize_text': {
                'description': 'Standardize text data formatting',
                'parameters': {
                    'columns': {
                        'type': 'list',
                        'description': 'Text columns to process',
                        'default': None,
                        'optional': True
                    },
                    'operations': {
                        'type': 'list',
                        'description': 'Text operations to apply',
                        'options': ['lowercase', 'uppercase', 'strip', 'remove_special_chars',
                                  'normalize_whitespace', 'remove_digits', 'title_case'],
                        'default': ['lowercase', 'strip', 'normalize_whitespace']
                    }
                },
                'use_cases': [
                    'Inconsistent text formatting',
                    'Leading/trailing whitespace',
                    'Mixed case text',
                    'Special characters cleanup'
                ]
            },
            
            'convert_data_types': {
                'description': 'Convert columns to appropriate data types',
                'parameters': {
                    'type_mapping': {
                        'type': 'dict',
                        'description': 'Mapping of column names to target types',
                        'example': {'col1': 'datetime', 'col2': 'numeric', 'col3': 'categorical'},
                        'required': True
                    }
                },
                'use_cases': [
                    'Incorrect data types after import',
                    'String numbers to numeric',
                    'Date strings to datetime',
                    'Categorical optimization'
                ]
            },
            
            'handle_categorical_data': {
                'description': 'Encode categorical variables',
                'parameters': {
                    'columns': {
                        'type': 'list',
                        'description': 'Categorical columns to encode',
                        'default': None,
                        'optional': True
                    },
                    'encoding': {
                        'type': 'string',
                        'description': 'Encoding method',
                        'options': ['label', 'onehot', 'frequency', 'target'],
                        'default': 'label'
                    },
                    'drop_original': {
                        'type': 'boolean',
                        'description': 'Whether to drop original columns',
                        'default': True
                    }
                },
                'use_cases': [
                    'Machine learning preprocessing',
                    'Categorical to numerical conversion',
                    'High cardinality categories',
                    'Ordinal encoding'
                ]
            },
            
            'normalize_numerical_data': {
                'description': 'Normalize or scale numerical data',
                'parameters': {
                    'columns': {
                        'type': 'list',
                        'description': 'Numerical columns to normalize',
                        'default': None,
                        'optional': True
                    },
                    'method': {
                        'type': 'string',
                        'description': 'Normalization method',
                        'options': ['standard', 'minmax', 'robust', 'log', 'sqrt'],
                        'default': 'standard'
                    }
                },
                'use_cases': [
                    'Different scales between features',
                    'Machine learning preprocessing',
                    'Skewed distributions',
                    'Algorithm requirements'
                ]
            },
            
            'validate_data_consistency': {
                'description': 'Validate data consistency based on rules',
                'parameters': {
                    'rules': {
                        'type': 'list',
                        'description': 'List of validation rules',
                        'example': [
                            {'type': 'range_check', 'column': 'age', 'min': 0, 'max': 120},
                            {'type': 'format_check', 'column': 'email', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
                        ],
                        'default': [],
                        'optional': True
                    }
                },
                'use_cases': [
                    'Business rule validation',
                    'Data quality checks',
                    'Format validation',
                    'Range validation'
                ]
            },
            
            'remove_constant_columns': {
                'description': 'Remove columns with constant or near-constant values',
                'parameters': {
                    'threshold': {
                        'type': 'float',
                        'description': 'Threshold for near-constant detection',
                        'default': 0.95,
                        'range': [0.8, 1.0]
                    }
                },
                'use_cases': [
                    'Constant value columns',
                    'Low variance features',
                    'Uninformative columns',
                    'Feature selection'
                ]
            },
            
            'handle_date_columns': {
                'description': 'Process date columns and extract features',
                'parameters': {
                    'date_columns': {
                        'type': 'list',
                        'description': 'Date columns to process',
                        'default': None,
                        'optional': True
                    },
                    'extract_features': {
                        'type': 'boolean',
                        'description': 'Whether to extract date features',
                        'default': True
                    }
                },
                'use_cases': [
                    'Date string parsing',
                    'Feature engineering from dates',
                    'Time series analysis',
                    'Seasonal pattern extraction'
                ]
            },
            
            'clean_column_names': {
                'description': 'Clean and standardize column names',
                'parameters': {},
                'use_cases': [
                    'Inconsistent column naming',
                    'Special characters in names',
                    'Mixed case column names',
                    'Spaces in column names'
                ]
            }
        }
    
    def get_strategy(self, strategy_name):
        """Get details of a specific strategy"""
        return self.strategies.get(strategy_name)
    
    def get_all_strategies(self):
        """Get all available strategies"""
        return self.strategies
    
    def get_strategies_by_use_case(self, use_case):
        """Get strategies that match a specific use case"""
        matching_strategies = []
        
        for strategy_name, strategy_info in self.strategies.items():
            if use_case.lower() in [uc.lower() for uc in strategy_info.get('use_cases', [])]:
                matching_strategies.append(strategy_name)
        
        return matching_strategies
    
    def get_strategy_recommendations(self, data_issues):
        """Get strategy recommendations based on data issues"""
        recommendations = []
        
        # Missing data recommendations
        if 'missing_data' in data_issues:
            missing_percentage = data_issues['missing_data'].get('percentage', 0)
            if missing_percentage > 50:
                recommendations.append({
                    'strategy': 'handle_missing_values',
                    'parameters': {'strategy': 'drop'},
                    'reason': 'High missing data percentage'
                })
            elif missing_percentage > 10:
                recommendations.append({
                    'strategy': 'handle_missing_values',
                    'parameters': {'strategy': 'fill_median'},
                    'reason': 'Moderate missing data percentage'
                })
        
        # Duplicate recommendations
        if 'duplicates' in data_issues:
            if data_issues['duplicates'].get('percentage', 0) > 0:
                recommendations.append({
                    'strategy': 'remove_duplicates',
                    'parameters': {'keep': 'first'},
                    'reason': 'Duplicate rows detected'
                })
        
        # Outlier recommendations
        if 'outliers' in data_issues:
            for column, outlier_info in data_issues['outliers'].items():
                if outlier_info.get('percentage', 0) > 5:
                    recommendations.append({
                        'strategy': 'handle_outliers',
                        'parameters': {'columns': [column], 'method': 'iqr'},
                        'reason': f'High outlier percentage in {column}'
                    })
        
        # Text standardization recommendations
        if 'text_issues' in data_issues:
            recommendations.append({
                'strategy': 'standardize_text',
                'parameters': {'operations': ['lowercase', 'strip', 'normalize_whitespace']},
                'reason': 'Text formatting inconsistencies'
            })
        
        # Data type recommendations
        if 'type_issues' in data_issues:
            type_mapping = {}
            for column, suggested_type in data_issues['type_issues'].items():
                type_mapping[column] = suggested_type
            
            if type_mapping:
                recommendations.append({
                    'strategy': 'convert_data_types',
                    'parameters': {'type_mapping': type_mapping},
                    'reason': 'Incorrect data types detected'
                })
        
        return recommendations
    
    def validate_parameters(self, strategy_name, parameters):
        """Validate parameters for a strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return False, f"Strategy '{strategy_name}' not found"
        
        strategy_params = strategy.get('parameters', {})
        errors = []
        
        # Check required parameters
        for param_name, param_info in strategy_params.items():
            if not param_info.get('optional', False) and param_name not in parameters:
                errors.append(f"Required parameter '{param_name}' missing")
        
        # Check parameter types and values
        for param_name, param_value in parameters.items():
            if param_name in strategy_params:
                param_info = strategy_params[param_name]
                
                # Check options
                if 'options' in param_info and param_value not in param_info['options']:
                    errors.append(f"Parameter '{param_name}' must be one of {param_info['options']}")
                
                # Check range
                if 'range' in param_info:
                    min_val, max_val = param_info['range']
                    if not (min_val <= param_value <= max_val):
                        errors.append(f"Parameter '{param_name}' must be between {min_val} and {max_val}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Parameters valid"
    
    def get_strategy_documentation(self, strategy_name):
        """Get comprehensive documentation for a strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        doc = {
            'name': strategy_name,
            'description': strategy['description'],
            'parameters': strategy.get('parameters', {}),
            'use_cases': strategy.get('use_cases', []),
            'examples': self._get_strategy_examples(strategy_name)
        }
        
        return doc
    
    def _get_strategy_examples(self, strategy_name):
        """Get examples for a strategy"""
        examples = {
            'handle_missing_values': [
                {
                    'description': 'Drop rows with missing values',
                    'parameters': {'strategy': 'drop'}
                },
                {
                    'description': 'Fill missing numerical values with mean',
                    'parameters': {'strategy': 'fill_mean'}
                },
                {
                    'description': 'Fill specific columns with median',
                    'parameters': {'columns': ['age', 'income'], 'strategy': 'fill_median'}
                }
            ],
            'remove_duplicates': [
                {
                    'description': 'Remove all duplicate rows, keep first occurrence',
                    'parameters': {'keep': 'first'}
                },
                {
                    'description': 'Remove duplicates based on specific columns',
                    'parameters': {'subset': ['name', 'email'], 'keep': 'first'}
                }
            ],
            'handle_outliers': [
                {
                    'description': 'Remove outliers using IQR method',
                    'parameters': {'method': 'iqr', 'threshold': 1.5}
                },
                {
                    'description': 'Cap outliers instead of removing',
                    'parameters': {'method': 'cap', 'threshold': 2.0}
                }
            ],
            'standardize_text': [
                {
                    'description': 'Basic text cleaning',
                    'parameters': {'operations': ['lowercase', 'strip', 'normalize_whitespace']}
                },
                {
                    'description': 'Aggressive text cleaning',
                    'parameters': {'operations': ['lowercase', 'strip', 'remove_special_chars', 'normalize_whitespace']}
                }
            ]
        }
        
        return examples.get(strategy_name, [])
    
    def get_cleaning_pipeline_template(self, data_profile):
        """Generate a cleaning pipeline template based on data profile"""
        pipeline = []
        
        # Step 1: Clean column names
        pipeline.append({
            'step': 1,
            'strategy': 'clean_column_names',
            'parameters': {},
            'reason': 'Standardize column names'
        })
        
        # Step 2: Handle missing values
        if data_profile.get('missing_data_percentage', 0) > 0:
            pipeline.append({
                'step': 2,
                'strategy': 'handle_missing_values',
                'parameters': {'strategy': 'fill_median'},
                'reason': 'Handle missing values'
            })
        
        # Step 3: Remove duplicates
        if data_profile.get('duplicate_percentage', 0) > 0:
            pipeline.append({
                'step': 3,
                'strategy': 'remove_duplicates',
                'parameters': {'keep': 'first'},
                'reason': 'Remove duplicate rows'
            })
        
        # Step 4: Handle outliers
        if data_profile.get('outlier_percentage', 0) > 5:
            pipeline.append({
                'step': 4,
                'strategy': 'handle_outliers',
                'parameters': {'method': 'iqr'},
                'reason': 'Handle outliers'
            })
        
        # Step 5: Standardize text
        if data_profile.get('text_columns', 0) > 0:
            pipeline.append({
                'step': 5,
                'strategy': 'standardize_text',
                'parameters': {'operations': ['lowercase', 'strip', 'normalize_whitespace']},
                'reason': 'Standardize text formatting'
            })
        
        # Step 6: Convert data types
        if data_profile.get('type_conversion_needed', False):
            pipeline.append({
                'step': 6,
                'strategy': 'convert_data_types',
                'parameters': {'type_mapping': data_profile.get('suggested_types', {})},
                'reason': 'Convert to appropriate data types'
            })
        
        return pipeline 