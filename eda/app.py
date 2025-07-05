#!/usr/bin/env python3
"""
AI-Powered EDA Analysis System

This Flask application analyzes datasets and generates intelligent visualizations
based on user prompts, returning everything in JSON format for frontend rendering.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
from dotenv import load_dotenv
from utils.data_analyzer import DataAnalyzer
from utils.business_intelligence import BusinessIntelligenceEngine
from utils.visualization_engine import VisualizationEngine

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif hasattr(obj, 'dtype'):
            return str(obj.dtype)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)

def safe_jsonify(data):
    """Safely convert data to JSON, handling numpy types"""
    return json.loads(json.dumps(data, cls=NumpyEncoder))

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

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])
    
    # Configuration - Increase limits for large datasets
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    app.config['UPLOAD_FOLDER'] = 'data'
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    
    # Additional configurations for handling large JSON payloads
    import sys
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(100000)
    
    # Initialize OpenAI client
    openai_client = None
    try:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {str(e)}. Will use fallback strategy.")
        openai_client = None
    
    # Initialize our engines
    viz_engine = VisualizationEngine()
    data_analyzer = DataAnalyzer()
    bi_engine = BusinessIntelligenceEngine()
    
    # Global variables to store current dataset and analysis
    current_dataset = None
    current_analysis = None
    
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'json'}
    
    def get_openai_analysis_strategy(df, user_prompt=None):
        """Use OpenAI to analyze dataset characteristics and create analysis strategy"""
        try:
            # Create a dataset summary
            dataset_summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_data': df.head(3).to_dict('records'),
                'missing_values': df.isnull().sum().to_dict(),
                'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
            }
            
            # Create prompt for OpenAI
            system_prompt = """You are an expert data analyst. Based on the dataset characteristics and user prompt, 
            recommend 8 specific visualizations (2 for each category):
            
            1. DESCRIPTIVE (What happened?): Show current state, distributions, summaries
            2. DIAGNOSTIC (Why did it happen?): Show correlations, comparisons, breakdowns
            3. PREDICTIVE (What will happen?): Show trends, forecasts, patterns
            4. PRESCRIPTIVE (What should we do?): Show recommendations, optimizations, actions
            
            Return a JSON response with exactly this structure:
            {
                "visualizations": [
                    {
                        "category": "descriptive|diagnostic|predictive|prescriptive",
                        "chart_type": "bar|line|scatter|pie|histogram|box|heatmap|area",
                        "x_axis": "column_name",
                        "y_axis": "column_name",
                        "title": "Chart Title",
                        "description": "Why this chart is useful",
                        "aggregation": "sum|mean|count|none"
                    }
                ]
            }"""
            
            user_message = f"""
            Dataset Info:
            - Shape: {dataset_summary['shape']}
            - Columns: {dataset_summary['columns']}
            - Data Types: {dataset_summary['dtypes']}
            - Numerical columns: {dataset_summary['numerical_columns']}
            - Categorical columns: {dataset_summary['categorical_columns']}
            - Sample data: {dataset_summary['sample_data']}
            
            User Prompt: {user_prompt or "Generate comprehensive analysis"}
            
            Please recommend 8 visualizations (2 per category) that would be most insightful for this dataset.
            """
            
            if openai_client:
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
            else:
                return get_fallback_strategy(df, user_prompt)
                
        except Exception as e:
            logger.error(f"Error getting OpenAI strategy: {str(e)}")
            return get_fallback_strategy(df, user_prompt)
    
    def get_fallback_strategy(df, user_prompt=None):
        """Fallback strategy when OpenAI is not available"""
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=['object']).columns)
        datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
        
        visualizations = []
        
        # Descriptive (2 charts)
        if len(numerical_cols) > 0:
            visualizations.append({
                "category": "descriptive",
                "chart_type": "histogram",
                "x_axis": numerical_cols[0],
                "y_axis": None,
                "title": f"Distribution of {numerical_cols[0]}",
                "description": "Shows the distribution of values",
                "aggregation": "none"
            })
        
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            visualizations.append({
                "category": "descriptive",
                "chart_type": "bar",
                "x_axis": categorical_cols[0],
                "y_axis": numerical_cols[0],
                "title": f"{numerical_cols[0]} by {categorical_cols[0]}",
                "description": "Shows values across categories",
                "aggregation": "mean"
            })
        
        # Diagnostic (2 charts)
        if len(numerical_cols) >= 2:
            visualizations.append({
                "category": "diagnostic",
                "chart_type": "scatter",
                "x_axis": numerical_cols[0],
                "y_axis": numerical_cols[1],
                "title": f"{numerical_cols[0]} vs {numerical_cols[1]}",
                "description": "Shows relationship between variables",
                "aggregation": "none"
            })
            
            visualizations.append({
                "category": "diagnostic",
                "chart_type": "heatmap",
                "x_axis": None,
                "y_axis": None,
                "title": "Correlation Matrix",
                "description": "Shows correlations between numerical variables",
                "aggregation": "none"
            })
        
        # Predictive (2 charts)
        if len(datetime_cols) > 0 and len(numerical_cols) > 0:
            visualizations.append({
                "category": "predictive",
                "chart_type": "line",
                "x_axis": datetime_cols[0],
                "y_axis": numerical_cols[0],
                "title": f"{numerical_cols[0]} Trend Over Time",
                "description": "Shows trend patterns for forecasting",
                "aggregation": "mean"
            })
        
        if len(numerical_cols) > 0:
            visualizations.append({
                "category": "predictive",
                "chart_type": "box",
                "x_axis": categorical_cols[0] if categorical_cols else None,
                "y_axis": numerical_cols[0],
                "title": f"{numerical_cols[0]} Distribution Analysis",
                "description": "Shows outliers and patterns for prediction",
                "aggregation": "none"
            })
        
        # Prescriptive (2 charts)
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            visualizations.append({
                "category": "prescriptive",
                "chart_type": "pie",
                "x_axis": categorical_cols[0],
                "y_axis": numerical_cols[0],
                "title": f"{numerical_cols[0]} Share by {categorical_cols[0]}",
                "description": "Shows areas for optimization",
                "aggregation": "sum"
            })
            
            visualizations.append({
                "category": "prescriptive",
                "chart_type": "bar",
                "x_axis": categorical_cols[0] if categorical_cols else numerical_cols[0],
                "y_axis": numerical_cols[-1] if len(numerical_cols) > 1 else numerical_cols[0],
                "title": "Performance Comparison",
                "description": "Shows areas for improvement",
                "aggregation": "mean"
            })
        
        return {"visualizations": visualizations}
    
    @app.route('/')
    def home():
        """Health check endpoint that returns JSON status"""
        return jsonify({
            'status': 'healthy',
            'service': 'AI-Powered EDA Analysis System',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/analyze', methods=['POST'])
    def analyze_data():
        """Main analysis endpoint - returns everything in JSON format"""
        nonlocal current_dataset, current_analysis
        
        try:
            df = None
            user_prompt = None
            
            # Handle file upload
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    
                    # Read the file based on extension
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file)
                    elif filename.endswith('.xlsx'):
                        df = pd.read_excel(file)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file)
                    
                    user_prompt = request.form.get('prompt', '')
                else:
                    return jsonify({'error': 'Invalid file format. Please upload CSV, XLSX, or JSON files.'}), 400
            
            # Handle JSON data
            elif request.is_json:
                data = request.get_json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    user_prompt = data.get('prompt', '')
                else:
                    return jsonify({'error': 'No data provided in JSON'}), 400
            
            else:
                return jsonify({'error': 'No file or data provided'}), 400
            
            if df is None or df.empty:
                return jsonify({'error': 'No valid data found'}), 400
            
            # Store current dataset
            current_dataset = df
            
            # Perform basic analysis
            analysis_results = data_analyzer.comprehensive_analysis(df)
            
            # Get OpenAI-powered visualization strategy
            viz_strategy = get_openai_analysis_strategy(df, user_prompt)
            
            # Generate visualizations as JSON data only
            visualizations = []
            
            for i, viz_config in enumerate(viz_strategy.get('visualizations', [])):
                try:
                    viz_result = viz_engine.create_visualization(df, viz_config)
                    if viz_result:
                        # Keep only the JSON chart data and metadata
                        chart_data = {
                            'id': f"chart_{i+1}",
                            'chart_type': viz_result.get('chart_type', ''),
                            'category': viz_config.get('category', ''),
                            'title': viz_config.get('title', f'Chart {i+1}'),
                            'description': viz_config.get('description', ''),
                            'chart_json': viz_result.get('chart_json', {}),
                            'config': viz_result.get('config', {}),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        visualizations.append(chart_data)
                        logger.info(f"Generated chart {i+1}: {viz_config.get('chart_type', 'unknown')}")
                        
                except Exception as e:
                    logger.error(f"Error creating visualization: {str(e)}")
                    continue
            
            # Generate business insights
            insights = bi_engine.generate_insights(df, user_prompt)
            
            # Store current analysis
            current_analysis = {
                'dataset_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                'analysis_results': analysis_results,
                'visualizations': visualizations,
                'insights': insights,
                'user_prompt': user_prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert to safe JSON format
            current_analysis = json_safe_convert(current_analysis)
            safe_analysis = safe_jsonify(current_analysis)
            
            return jsonify({
                'status': 'success',
                'message': 'Analysis completed successfully',
                'analysis': safe_analysis
            })
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/visualizations')
    def get_visualizations():
        """Get all generated visualizations"""
        if current_analysis and 'visualizations' in current_analysis:
            return jsonify({
                'status': 'success',
                'visualizations': safe_jsonify(current_analysis['visualizations'])
            })
        else:
            return jsonify({'error': 'No visualizations available. Please analyze data first.'}), 404
    
    @app.route('/insights')
    def get_insights():
        """Get business insights"""
        if current_analysis and 'insights' in current_analysis:
            return jsonify({
                'status': 'success',
                'insights': safe_jsonify(current_analysis['insights'])
            })
        else:
            return jsonify({'error': 'No insights available. Please analyze data first.'}), 404
    
    @app.route('/export')
    def export_analysis():
        """Export complete analysis results"""
        if current_analysis:
            return jsonify({
                'status': 'success',
                'analysis': safe_jsonify(current_analysis)
            })
        else:
            return jsonify({'error': 'No analysis available. Please analyze data first.'}), 404
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 3035))
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"🚀 AI EDA Analysis System is running on port {port}") 