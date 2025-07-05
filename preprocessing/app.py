from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import openai
from openai import OpenAI
import traceback
import io
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.data_preprocessor import DataPreprocessor
from utils.preprocessing_analyzer import PreprocessingAnalyzer
from utils.standardization_techniques import StandardizationTechniques

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class AIDataPreprocessingAgent:
    def __init__(self):
        self.analyzer = PreprocessingAnalyzer()
        self.preprocessor = DataPreprocessor()
        self.techniques = StandardizationTechniques()
        
    def analyze_dataset(self, df):
        """Analyze the dataset for preprocessing needs"""
        analysis = self.analyzer.comprehensive_analysis(df)
        return analysis
    
    def get_preprocessing_recommendations(self, analysis):
        """Use OpenAI to decide on preprocessing strategies based on analysis"""
        
        # Prepare the prompt for OpenAI
        prompt = self._create_preprocessing_prompt(analysis)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist specializing in data preprocessing, standardization, and normalization. Analyze the provided dataset information and recommend appropriate preprocessing techniques to prepare the data for machine learning or analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            recommendations = response.choices[0].message.content
            return self._parse_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}")
            # Fallback to rule-based recommendations
            return self._fallback_recommendations(analysis)
    
    def _create_preprocessing_prompt(self, analysis):
        """Create a detailed prompt for OpenAI based on dataset analysis"""
        prompt = f"""
        I need you to analyze this dataset and recommend specific data preprocessing, standardization, and normalization techniques.
        
        DATASET OVERVIEW:
        - Shape: {analysis['basic_info']['shape']}
        - Columns: {analysis['basic_info']['columns']}
        - Data Types: {json.dumps(analysis['basic_info']['dtypes'], indent=2)}
        
        NUMERICAL COLUMNS ANALYSIS:
        {json.dumps(analysis['numerical_analysis'], indent=2)}
        
        CATEGORICAL COLUMNS ANALYSIS:
        {json.dumps(analysis['categorical_analysis'], indent=2)}
        
        DISTRIBUTION ANALYSIS:
        {json.dumps(analysis['distribution_analysis'], indent=2)}
        
        SCALING REQUIREMENTS:
        {json.dumps(analysis['scaling_requirements'], indent=2)}
        
        ENCODING REQUIREMENTS:
        {json.dumps(analysis['encoding_requirements'], indent=2)}
        
        FEATURE ENGINEERING OPPORTUNITIES:
        {json.dumps(analysis['feature_engineering'], indent=2)}
        
        Please recommend specific preprocessing techniques from the following available options:
        
        STANDARDIZATION TECHNIQUES:
        1. standardize_numerical_features (methods: z_score, min_max, robust, quantile_uniform, quantile_normal)
        2. normalize_categorical_features (methods: label_encoding, one_hot_encoding, target_encoding, frequency_encoding, binary_encoding)
        3. handle_text_features (methods: tfidf, count_vectorizer, word_embeddings, text_preprocessing)
        4. engineer_datetime_features (extract: year, month, day, hour, weekday, quarter, is_weekend, days_since_epoch)
        5. create_polynomial_features (degree: 2, 3, interaction_only, include_bias)
        6. apply_feature_selection (methods: variance_threshold, correlation_threshold, mutual_info, chi2, f_classif)
        7. handle_imbalanced_data (methods: smote, adasyn, random_oversample, random_undersample, tomek_links)
        8. detect_and_handle_outliers (methods: isolation_forest, local_outlier_factor, one_class_svm, elliptic_envelope)
        9. apply_dimensionality_reduction (methods: pca, lda, tsne, umap, factor_analysis)
        10. create_interaction_features (methods: polynomial, multiplicative, additive, ratio_features)
        
        Return your recommendations in the following JSON format:
        {{
            "preprocessing_pipeline": [
                {{
                    "technique": "technique_name",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "reason": "explanation for this technique",
                    "priority": 1-10,
                    "columns": ["column1", "column2"] or "all" or "numerical" or "categorical"
                }}
            ],
            "expected_improvements": "description of expected improvements",
            "preprocessing_strategy": "overall strategy (e.g., 'machine_learning_prep', 'analysis_prep', 'visualization_prep')",
            "potential_considerations": "any important considerations or warnings"
        }}
        
        Focus on techniques that will make the data ready for machine learning models, improve data quality, and extract meaningful features.
        Consider the data distribution, scale differences, categorical encoding needs, and feature engineering opportunities.
        """
        return prompt
    
    def _parse_recommendations(self, recommendations_text):
        """Parse AI recommendations into structured format"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', recommendations_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                recommendations = json.loads(json_str)
                return recommendations
            else:
                # If no JSON found, create a simple structure
                return {
                    "preprocessing_pipeline": [],
                    "expected_improvements": recommendations_text,
                    "preprocessing_strategy": "general_preprocessing",
                    "potential_considerations": "Could not parse specific recommendations"
                }
        except Exception as e:
            logger.error(f"Error parsing recommendations: {str(e)}")
            return self._fallback_recommendations({})
    
    def _fallback_recommendations(self, analysis):
        """Fallback rule-based recommendations if AI fails"""
        pipeline = []
        
        # Basic preprocessing steps based on analysis
        numerical_cols = analysis.get('numerical_analysis', {}).keys()
        categorical_cols = analysis.get('categorical_analysis', {}).keys()
        
        if numerical_cols:
            pipeline.append({
                "technique": "standardize_numerical_features",
                "parameters": {"method": "z_score"},
                "reason": "Standardize numerical features for better model performance",
                "priority": 8,
                "columns": list(numerical_cols)
            })
        
        if categorical_cols:
            pipeline.append({
                "technique": "normalize_categorical_features",
                "parameters": {"method": "one_hot_encoding"},
                "reason": "Encode categorical features for machine learning",
                "priority": 7,
                "columns": list(categorical_cols)
            })
        
        return {
            "preprocessing_pipeline": pipeline,
            "expected_improvements": "Basic preprocessing applied",
            "preprocessing_strategy": "general_preprocessing",
            "potential_considerations": "Conservative approach with standard techniques"
        }
    
    def execute_preprocessing(self, df, recommendations):
        """Execute the recommended preprocessing steps"""
        processed_df = df.copy()
        execution_log = []
        
        # Sort steps by priority (higher priority first)
        pipeline = sorted(recommendations.get('preprocessing_pipeline', []), 
                         key=lambda x: x.get('priority', 0), reverse=True)
        
        for step in pipeline:
            try:
                technique_name = step['technique']
                parameters = step.get('parameters', {})
                columns = step.get('columns', 'all')
                
                # Execute the preprocessing technique
                if hasattr(self.preprocessor, technique_name):
                    method = getattr(self.preprocessor, technique_name)
                    processed_df = method(processed_df, columns=columns, **parameters)
                    
                    execution_log.append({
                        "step": technique_name,
                        "parameters": parameters,
                        "columns": columns,
                        "shape_after": processed_df.shape,
                        "status": "success",
                        "reason": step.get('reason', 'No reason provided')
                    })
                    
                    logger.info(f"Successfully executed {technique_name}")
                    
                else:
                    execution_log.append({
                        "step": technique_name,
                        "parameters": parameters,
                        "columns": columns,
                        "status": "failed",
                        "error": f"Method {technique_name} not found",
                        "reason": step.get('reason', 'No reason provided')
                    })
                    
                    logger.error(f"Method {technique_name} not found in preprocessor")
                    
            except Exception as e:
                execution_log.append({
                    "step": technique_name,
                    "parameters": parameters,
                    "columns": columns,
                    "status": "failed",
                    "error": str(e),
                    "reason": step.get('reason', 'No reason provided')
                })
                
                logger.error(f"Error executing {technique_name}: {str(e)}")
        
        return processed_df, execution_log

# Initialize the AI agent
agent = AIDataPreprocessingAgent()

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'service': 'AI Data Preprocessing Agent',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded CSV data for preprocessing needs"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Analyze the dataset
        analysis = agent.analyze_dataset(df)
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset analyzed successfully',
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess data using AI recommendations"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        original_shape = df.shape
        
        # Analyze the dataset
        analysis = agent.analyze_dataset(df)
        
        # Get AI recommendations
        recommendations = agent.get_preprocessing_recommendations(analysis)
        
        # Execute preprocessing
        processed_df, execution_log = agent.execute_preprocessing(df, recommendations)
        
        # Save processed data
        output_file = DATA_DIR / "preprocessed.csv"
        processed_df.to_csv(output_file, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'Data preprocessed successfully',
            'original_shape': original_shape,
            'processed_shape': processed_df.shape,
            'recommendations': recommendations,
            'execution_log': execution_log,
            'output_file': str(output_file)
        })
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/preprocessed', methods=['GET'])
def download_preprocessed():
    """Download the preprocessed CSV file"""
    try:
        output_file = DATA_DIR / "preprocessed.csv"
        if output_file.exists():
            return send_file(output_file, as_attachment=True, download_name='preprocessed.csv')
        else:
            return jsonify({'error': 'No preprocessed file available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/techniques', methods=['GET'])
def get_available_techniques():
    """Get all available preprocessing techniques"""
    try:
        techniques = agent.techniques.get_all_techniques()
        return jsonify({
            'status': 'success',
            'techniques': techniques
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/custom-preprocess', methods=['POST'])
def custom_preprocess():
    """Preprocess data with custom parameters"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get custom preprocessing configuration
        config = request.form.get('config')
        if config:
            config = json.loads(config)
        else:
            return jsonify({'error': 'No preprocessing configuration provided'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        original_shape = df.shape
        
        # Execute custom preprocessing
        processed_df, execution_log = agent.execute_preprocessing(df, config)
        
        # Save processed data
        output_file = DATA_DIR / "custom_preprocessed.csv"
        processed_df.to_csv(output_file, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'Custom preprocessing completed',
            'original_shape': original_shape,
            'processed_shape': processed_df.shape,
            'execution_log': execution_log,
            'output_file': str(output_file)
        })
        
    except Exception as e:
        logger.error(f"Error in custom preprocessing: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set. AI recommendations will fall back to rule-based approach.")
    
    port = int(os.environ.get('PORT', 3031))
    app.run(host='0.0.0.0', port=port, debug=True) 