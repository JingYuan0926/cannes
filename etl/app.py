from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback
import io
import tempfile
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

from utils.data_analyzer import DataAnalyzer
from utils.data_cleaner import DataCleaner
from utils.cleaning_strategies import CleaningStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client with error handling
client = None
try:
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")
except Exception as e:
    logger.warning(f"Failed to initialize OpenAI client: {str(e)}. Will use fallback recommendations.")
    client = None

# Global variables
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def json_safe_convert(obj):
    """Convert data to JSON-safe format, handling NaN values"""
    if isinstance(obj, dict):
        return {key: json_safe_convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_convert(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

class AIDataCleaningAgent:
    def __init__(self):
        self.analyzer = DataAnalyzer()
        self.cleaner = DataCleaner()
        self.strategies = CleaningStrategies()
        
    def analyze_dataset(self, df):
        """Analyze the dataset and generate insights for AI decision making"""
        analysis = self.analyzer.comprehensive_analysis(df)
        return analysis
    
    def get_cleaning_recommendations(self, analysis):
        """Use OpenAI to decide on cleaning strategies based on analysis"""
        
        if client is None:
            logger.info("OpenAI client not available, using fallback recommendations")
            return self._fallback_recommendations(analysis)
        
        # Prepare the prompt for OpenAI
        prompt = self._create_analysis_prompt(analysis)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist specializing in data cleaning and preprocessing. Analyze the provided dataset information and recommend appropriate cleaning strategies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            recommendations = response.choices[0].message.content
            return self._parse_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}")
            # Fallback to rule-based recommendations
            return self._fallback_recommendations(analysis)
    
    def _create_analysis_prompt(self, analysis):
        """Create a detailed prompt for OpenAI based on dataset analysis"""
        prompt = f"""
        I need you to analyze this dataset and recommend specific data cleaning strategies.
        
        DATASET OVERVIEW:
        - Shape: {analysis['basic_info']['shape']}
        - Columns: {analysis['basic_info']['columns']}
        - Data Types: {json.dumps(analysis['basic_info']['dtypes'], indent=2)}
        
        MISSING DATA:
        {json.dumps(analysis['missing_data'], indent=2)}
        
        DUPLICATES:
        - Total duplicates: {analysis['duplicates']['total_duplicates']}
        - Duplicate percentage: {analysis['duplicates']['duplicate_percentage']:.2f}%
        
        OUTLIERS:
        {json.dumps(analysis['outliers'], indent=2)}
        
        DATA QUALITY ISSUES:
        {json.dumps(analysis['data_quality'], indent=2)}
        
        STATISTICAL SUMMARY:
        {analysis['statistical_summary']}
        
        Please recommend specific cleaning strategies from the following available options:
        1. handle_missing_values (strategies: drop, fill_mean, fill_median, fill_mode, forward_fill, backward_fill, interpolate)
        2. remove_duplicates (keep: first, last, False)
        3. handle_outliers (method: iqr, z_score, isolation_forest, percentile)
        4. standardize_text (operations: lowercase, strip, remove_special_chars, normalize_whitespace)
        5. convert_data_types (specify target types)
        6. handle_categorical_data (encoding: label, onehot, target)
        7. normalize_numerical_data (method: standard, minmax, robust)
        8. validate_data_consistency (rules for consistency checks)
        
        Return your recommendations in the following JSON format:
        {{
            "cleaning_steps": [
                {{
                    "method": "method_name",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "reason": "explanation for this step",
                    "priority": 1-10
                }}
            ],
            "expected_improvements": "description of expected improvements",
            "potential_risks": "any potential risks or data loss warnings"
        }}
        
        Focus on the most critical issues first and provide practical, executable recommendations.
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
                    "cleaning_steps": [],
                    "expected_improvements": recommendations_text,
                    "potential_risks": "Could not parse specific recommendations"
                }
        except Exception as e:
            logger.error(f"Error parsing recommendations: {str(e)}")
            return self._fallback_recommendations({})
    
    def _fallback_recommendations(self, analysis):
        """Fallback rule-based recommendations if AI fails"""
        steps = []
        
        # Basic cleaning steps based on analysis
        if analysis.get('duplicates', {}).get('total_duplicates', 0) > 0:
            steps.append({
                "method": "remove_duplicates",
                "parameters": {"keep": "first"},
                "reason": "Remove duplicate rows",
                "priority": 8
            })
        
        missing_data = analysis.get('missing_data', {})
        if missing_data:
            for col, info in missing_data.items():
                if info.get('missing_percentage', 0) > 0:
                    steps.append({
                        "method": "handle_missing_values",
                        "parameters": {"columns": [col], "strategy": "drop" if info['missing_percentage'] > 50 else "fill_mean"},
                        "reason": f"Handle missing values in {col}",
                        "priority": 9
                    })
        
        return {
            "cleaning_steps": steps,
            "expected_improvements": "Basic cleaning applied",
            "potential_risks": "Minimal risk with conservative approach"
        }
    
    def execute_cleaning(self, df, recommendations):
        """Execute the recommended cleaning steps"""
        cleaned_df = df.copy()
        execution_log = []
        
        # Sort steps by priority (higher priority first)
        steps = sorted(recommendations.get('cleaning_steps', []), 
                      key=lambda x: x.get('priority', 0), reverse=True)
        
        for step in steps:
            try:
                method_name = step['method']
                parameters = step.get('parameters', {})
                
                # Execute the cleaning method
                if hasattr(self.cleaner, method_name):
                    method = getattr(self.cleaner, method_name)
                    cleaned_df = method(cleaned_df, **parameters)
                    
                    execution_log.append({
                        "step": method_name,
                        "parameters": parameters,
                        "status": "success",
                        "reason": step.get('reason', ''),
                        "shape_after": cleaned_df.shape
                    })
                    
                    logger.info(f"Executed {method_name} with parameters {parameters}")
                else:
                    execution_log.append({
                        "step": method_name,
                        "parameters": parameters,
                        "status": "failed",
                        "error": f"Method {method_name} not found",
                        "reason": step.get('reason', '')
                    })
                    
            except Exception as e:
                execution_log.append({
                    "step": step['method'],
                    "parameters": step.get('parameters', {}),
                    "status": "failed",
                    "error": str(e),
                    "reason": step.get('reason', '')
                })
                logger.error(f"Error executing {step['method']}: {str(e)}")
        
        return cleaned_df, execution_log

# Initialize the AI agent
ai_agent = AIDataCleaningAgent()

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Data Cleaning Agent (ETL)',
        'timestamp': datetime.now().isoformat(),
        'openai_available': client is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_and_clean():
    """Main endpoint that combines analysis and cleaning for the pipeline"""
    try:
        # Handle both file upload and JSON data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            df = pd.read_csv(file)
        elif request.is_json:
            data = request.get_json()
            if 'data' not in data:
                return jsonify({'error': 'No data provided in JSON'}), 400
            df = pd.DataFrame(data['data'])
        else:
            return jsonify({'error': 'No file or data provided'}), 400
        
        goal = request.form.get('goal') if 'file' in request.files else request.get_json().get('goal', '')
        original_shape = df.shape
        
        # Analyze the dataset
        logger.info("Analyzing dataset...")
        analysis = ai_agent.analyze_dataset(df)
        
        # Get AI recommendations
        logger.info("Getting cleaning recommendations...")
        recommendations = ai_agent.get_cleaning_recommendations(analysis)
        
        # Execute cleaning
        logger.info("Executing cleaning steps...")
        cleaned_df, execution_log = ai_agent.execute_cleaning(df, recommendations)
        
        # Convert cleaned dataframe to list of dictionaries for JSON serialization
        # Handle NaN values which are not valid JSON
        cleaned_df_for_json = cleaned_df.fillna('')  # Replace NaN with empty string
        processed_data = cleaned_df_for_json.to_dict('records')
        
        # Ensure all data is JSON-safe
        analysis = json_safe_convert(analysis)
        recommendations = json_safe_convert(recommendations)
        execution_log = json_safe_convert(execution_log)
        
        return jsonify({
            'status': 'success',
            'message': 'ETL process completed successfully',
            'original_shape': original_shape,
            'cleaned_shape': cleaned_df.shape,
            'processed_data': processed_data,
            'analysis': analysis,
            'recommendations': recommendations,
            'execution_log': execution_log,
            'goal': goal
        })
        
    except Exception as e:
        logger.error(f"Error in ETL process: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/cleaned', methods=['GET'])
def download_cleaned():
    """Download the cleaned CSV file"""
    try:
        output_path = DATA_DIR / "cleaned.csv"
        if output_path.exists():
            return send_file(output_path, as_attachment=True, download_name='cleaned.csv')
        else:
            return jsonify({'error': 'Cleaned file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/strategies', methods=['GET'])
def get_available_strategies():
    """Get all available cleaning strategies"""
    try:
        strategies = ai_agent.strategies.get_all_strategies()
        return jsonify({
            'status': 'success',
            'strategies': strategies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/custom-clean', methods=['POST'])
def custom_clean():
    """Clean data with custom parameters"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get custom cleaning steps from request
        custom_steps = request.form.get('steps')
        if custom_steps:
            custom_steps = json.loads(custom_steps)
        else:
            return jsonify({'error': 'No cleaning steps provided'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        original_shape = df.shape
        
        # Execute custom cleaning
        recommendations = {'cleaning_steps': custom_steps}
        cleaned_df, execution_log = ai_agent.execute_cleaning(df, recommendations)
        
        # Save cleaned data
        output_path = DATA_DIR / "cleaned.csv"
        cleaned_df.to_csv(output_path, index=False)
        
        return jsonify({
            'status': 'success',
            'original_shape': original_shape,
            'cleaned_shape': cleaned_df.shape,
            'execution_log': execution_log,
            'output_file': str(output_path),
            'message': 'Custom cleaning completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in custom cleaning: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set. AI recommendations will fall back to rule-based approach.")
    
    port = int(os.environ.get('PORT', 3030))
    logger.info(f"Starting ETL service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 