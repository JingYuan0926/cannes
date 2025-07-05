#!/usr/bin/env python3
"""
AI-Powered Data Analysis System

This Flask application performs intelligent data analysis using various algorithms
based on user goals and dataset characteristics. It uses OpenAI to determine the
best analysis approach and generates comprehensive insights with visualizations.

Analytics Categories:
1. Descriptive Analytics (What happened?) - Clustering, Dimensionality Reduction
2. Predictive Analytics (What will happen?) - Regression, Classification
3. Prescriptive Analytics (What should we do?) - Optimization, Recommendations
4. Diagnostic Analytics (Why did it happen?) - Feature Importance, Causal Analysis
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import openai
from datetime import datetime
import logging
from dotenv import load_dotenv
from utils.descriptive_analytics import DescriptiveAnalytics
from utils.predictive_analytics import PredictiveAnalytics
from utils.prescriptive_analytics import PrescriptiveAnalytics
from utils.diagnostic_analytics import DiagnosticAnalytics
from utils.analysis_orchestrator import AnalysisOrchestrator

# Load environment variables
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
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def safe_jsonify(data):
    """Safely convert data to JSON using custom encoder"""
    def clean_data(obj):
        """Recursively clean data to handle NaN values"""
        if isinstance(obj, dict):
            return {key: clean_data(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif pd.isna(obj):
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    cleaned_data = clean_data(data)
    return json.loads(json.dumps(cleaned_data, cls=NumpyEncoder))

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['UPLOAD_FOLDER'] = 'data'
    app.json_encoder = NumpyEncoder  # Set custom encoder for Flask
    
    # Initialize OpenAI
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize analysis orchestrator
    orchestrator = AnalysisOrchestrator()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'json', 'parquet'}
    
    @app.route('/')
    def home():
        """Health check and API information"""
        response_data = {
            'service': 'AI-Powered Data Analysis System',
            'status': 'running',
            'version': '1.0.0',
            'endpoints': {
                '/analyze': 'POST - Main analysis endpoint',
                '/health': 'GET - Health check',
                '/results/<analysis_id>': 'GET - Retrieve analysis results'
            },
            'analytics_types': [
                'Descriptive Analytics (Clustering, Dimensionality Reduction)',
                'Predictive Analytics (Regression, Classification)',
                'Prescriptive Analytics (Optimization, Recommendations)',
                'Diagnostic Analytics (Feature Importance, Causal Analysis)'
            ],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(safe_jsonify(response_data))
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        response_data = {
            'status': 'healthy',
            'service': 'AI Data Analysis System',
            'openai_configured': bool(openai.api_key),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(safe_jsonify(response_data))
    
    @app.route('/analyze', methods=['POST'])
    def analyze_data():
        """
        Main analysis endpoint that accepts dataset and goal, then performs AI-powered analysis
        
        Expected input:
        - File upload (CSV, XLSX, JSON, Parquet) + goal parameter
        - JSON payload with data and goal
        
        Returns comprehensive analysis results with graphs and insights
        """
        try:
            df = None
            goal = None
            
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
                    elif filename.endswith('.parquet'):
                        df = pd.read_parquet(file)
                    
                    goal = request.form.get('goal', '')
                else:
                    return jsonify({'error': 'Invalid file format. Please upload CSV, XLSX, JSON, or Parquet files.'}), 400
            
            # Handle JSON data
            elif request.is_json:
                data = request.get_json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    goal = data.get('goal', '')
                    
                    # Convert data types - preprocessing may return everything as strings
                    logger.info("Converting data types for analysis...")
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Try to convert to numeric first
                            try:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                            
                            # Try to convert to datetime if it looks like dates
                            if df[col].dtype == 'object':
                                try:
                                    if df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
                                        df[col] = pd.to_datetime(df[col], errors='ignore')
                                except:
                                    pass
                    
                    logger.info(f"After type conversion - dtypes: {df.dtypes.to_dict()}")
                else:
                    return jsonify({'error': 'No data provided in JSON'}), 400
            
            else:
                return jsonify({'error': 'No file or data provided'}), 400
            
            if df is None or df.empty:
                return jsonify({'error': 'No valid data found'}), 400
            
            if not goal:
                return jsonify({'error': 'Goal parameter is required'}), 400
            
            # Perform comprehensive AI-powered analysis
            logger.info(f"Starting analysis for goal: {goal}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Dataset dtypes: {df.dtypes.to_dict()}")
            logger.info(f"Sample data: {df.head(2).to_dict('records')}")
            
            analysis_results = orchestrator.perform_comprehensive_analysis(df, goal)
            
            logger.info(f"Analysis results received - analyses: {len(analysis_results.get('analyses', []))}, graphs: {len(analysis_results.get('graphs', []))}")
            
            if 'error' in analysis_results:
                logger.error(f"Analysis error: {analysis_results['error']}")
                return jsonify(safe_jsonify(analysis_results)), 500
            
            analysis_id = analysis_results['analysis_id']
            
            logger.info(f"Analysis completed successfully for analysis_id: {analysis_id}")
            
            response_data = {
                'status': 'success',
                'message': 'Analysis completed successfully',
                'analysis_id': analysis_id,
                'summary': {
                    'total_analyses': len(analysis_results.get('analyses', [])),
                    'total_graphs': len(analysis_results.get('graphs', [])),
                    'analytics_types': list(analysis_results.get('analytics_summary', {}).keys())
                },
                'results': analysis_results
            }
            
            return jsonify(safe_jsonify(response_data))
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            error_response = {'error': f'Analysis failed: {str(e)}'}
            return jsonify(safe_jsonify(error_response)), 500
    
    @app.route('/results/<analysis_id>')
    def get_analysis_results(analysis_id):
        """Retrieve analysis results by ID"""
        try:
            results_file = f"results/analysis_{analysis_id}.json"
            
            if not os.path.exists(results_file):
                return jsonify({'error': 'Analysis results not found'}), 404
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            response_data = {
                'status': 'success',
                'analysis_id': analysis_id,
                'results': results
            }
            
            return jsonify(safe_jsonify(response_data))
            
        except Exception as e:
            logger.error(f"Error retrieving results: {str(e)}")
            error_response = {'error': f'Failed to retrieve results: {str(e)}'}
            return jsonify(safe_jsonify(error_response)), 500
    
    @app.route('/list-analyses')
    def list_analyses():
        """List all available analysis results"""
        try:
            results_dir = 'results'
            if not os.path.exists(results_dir):
                return jsonify({'analyses': []})
            
            analyses = []
            for filename in os.listdir(results_dir):
                if filename.startswith('analysis_') and filename.endswith('.json'):
                    analysis_id = filename.replace('analysis_', '').replace('.json', '')
                    filepath = os.path.join(results_dir, filename)
                    
                    # Get basic info
                    stat = os.stat(filepath)
                    
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            goal = data.get('goal', 'Unknown')
                            timestamp = data.get('timestamp', 'Unknown')
                    except:
                        goal = 'Unknown'
                        timestamp = 'Unknown'
                    
                    analyses.append({
                        'analysis_id': analysis_id,
                        'goal': goal,
                        'timestamp': timestamp,
                        'file_size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            analyses.sort(key=lambda x: x['created'], reverse=True)
            
            response_data = {
                'status': 'success',
                'total_analyses': len(analyses),
                'analyses': analyses
            }
            
            return jsonify(safe_jsonify(response_data))
            
        except Exception as e:
            logger.error(f"Error listing analyses: {str(e)}")
            error_response = {'error': f'Failed to list analyses: {str(e)}'}
            return jsonify(safe_jsonify(error_response)), 500
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 3040))
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"🚀 AI Data Analysis System is running on port {port}") 