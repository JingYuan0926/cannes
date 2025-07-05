#!/usr/bin/env python3
"""
AI Data Analysis Pipeline

A comprehensive Flask application that uses OpenAI to determine appropriate analysis
techniques based on user prompts and dataset characteristics. Provides business
intelligence through descriptive, diagnostic, prescriptive, and predictive analytics.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
from dotenv import load_dotenv
import openai
from io import BytesIO
import base64
import traceback

# Import our utility modules
from utils.data_analyzer import DataAnalyzer
from utils.visualization_engine import VisualizationEngine
from utils.filter_engine import FilterEngine
from utils.business_intelligence import BusinessIntelligenceEngine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global variables for storing analysis results
current_dataset = None
current_analysis = None
current_visualizations = None

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
PORT = int(os.getenv('PORT', 3032))
HOST = os.getenv('HOST', '0.0.0.0')

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_openai_analysis_recommendations(df, user_prompt=None):
    """Use OpenAI to determine appropriate analysis techniques"""
    try:
        # Prepare dataset summary for OpenAI
        dataset_summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(3).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Create prompt for OpenAI
        system_prompt = """You are a senior data analyst and business intelligence expert. 
        Analyze the provided dataset and user request to recommend appropriate analysis techniques.
        
        Your response should be a JSON object with the following structure:
        {
            "analysis_recommendations": [
                {
                    "type": "descriptive|diagnostic|prescriptive|predictive",
                    "technique": "specific_technique_name",
                    "description": "what this analysis will show",
                    "columns": ["column1", "column2"],
                    "filters": {"column": "value"},
                    "visualization": "chart_type",
                    "business_value": "business insight this provides",
                    "priority": 1-10
                }
            ],
            "key_insights": ["insight1", "insight2"],
            "business_questions": ["question1", "question2"],
            "recommended_visualizations": [
                {
                    "chart_type": "bar|line|scatter|pie|heatmap|box|histogram",
                    "x_axis": "column_name",
                    "y_axis": "column_name",
                    "group_by": "column_name",
                    "title": "Chart Title",
                    "business_context": "why this chart is valuable"
                }
            ]
        }
        
        Focus on:
        1. Business intelligence and actionable insights
        2. Trend analysis and patterns
        3. Performance metrics and KPIs
        4. Comparative analysis
        5. Time-series analysis if datetime columns exist
        6. Segmentation and grouping analysis
        7. Correlation and relationship analysis
        """
        
        user_content = f"""
        Dataset Summary:
        {json.dumps(dataset_summary, indent=2, default=str)}
        
        User Request: {user_prompt if user_prompt else "Perform comprehensive business intelligence analysis"}
        
        Please recommend appropriate analysis techniques and visualizations for this dataset.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse OpenAI response
        recommendations = json.loads(response.choices[0].message.content)
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting OpenAI recommendations: {str(e)}")
        # Return fallback recommendations
        return {
            "analysis_recommendations": [
                {
                    "type": "descriptive",
                    "technique": "basic_statistics",
                    "description": "Basic statistical summary of the dataset",
                    "columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "filters": {},
                    "visualization": "histogram",
                    "business_value": "Understanding data distribution and basic patterns",
                    "priority": 8
                }
            ],
            "key_insights": ["Dataset contains " + str(df.shape[0]) + " records"],
            "business_questions": ["What are the key patterns in this data?"],
            "recommended_visualizations": []
        }

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Data Analysis Pipeline',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_dataset():
    """Analyze dataset and generate AI-powered recommendations"""
    global current_dataset, current_analysis
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get user prompt if provided
        user_prompt = request.form.get('prompt', '')
        
        # Read the dataset
        filename = secure_filename(file.filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Store current dataset
        current_dataset = df.copy()
        
        # Initialize analysis engines
        data_analyzer = DataAnalyzer()
        bi_engine = BusinessIntelligenceEngine()
        
        # Get OpenAI recommendations
        ai_recommendations = get_openai_analysis_recommendations(df, user_prompt)
        
        # Perform basic data analysis
        basic_analysis = data_analyzer.comprehensive_analysis(df)
        
        # Generate business intelligence insights
        bi_insights = bi_engine.generate_insights(df, user_prompt)
        
        # Combine all analysis results
        current_analysis = {
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'ai_recommendations': ai_recommendations,
            'basic_analysis': basic_analysis,
            'business_intelligence': bi_insights,
            'user_prompt': user_prompt,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset analyzed successfully',
            'analysis': current_analysis
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def generate_visualizations():
    """Generate visualizations based on analysis recommendations"""
    global current_dataset, current_analysis, current_visualizations
    
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded. Please analyze a dataset first.'}), 400
        
        # Get visualization parameters
        data = request.get_json()
        chart_configs = data.get('chart_configs', [])
        filters = data.get('filters', {})
        
        # Initialize visualization engine
        viz_engine = VisualizationEngine()
        filter_engine = FilterEngine()
        
        # Apply filters if provided
        filtered_df = filter_engine.apply_filters(current_dataset, filters)
        
        # Generate visualizations
        visualizations = []
        
        # If no specific chart configs provided, use AI recommendations
        if not chart_configs and current_analysis:
            chart_configs = current_analysis['ai_recommendations'].get('recommended_visualizations', [])
        
        for config in chart_configs:
            try:
                chart_data = viz_engine.create_visualization(filtered_df, config)
                if chart_data:
                    visualizations.append(chart_data)
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}")
                continue
        
        # Store current visualizations
        current_visualizations = visualizations
        
        return jsonify({
            'status': 'success',
            'message': f'Generated {len(visualizations)} visualizations',
            'visualizations': visualizations,
            'filtered_data_shape': filtered_df.shape
        })
        
    except Exception as e:
        logger.error(f"Error in generate_visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Visualization generation failed: {str(e)}'}), 500

@app.route('/insights', methods=['GET'])
def get_insights():
    """Get comprehensive business intelligence insights"""
    global current_dataset, current_analysis
    
    try:
        if current_dataset is None or current_analysis is None:
            return jsonify({'error': 'No analysis available. Please analyze a dataset first.'}), 400
        
        # Initialize BI engine
        bi_engine = BusinessIntelligenceEngine()
        
        # Generate comprehensive insights
        insights = bi_engine.generate_comprehensive_insights(
            current_dataset, 
            current_analysis
        )
        
        return jsonify({
            'status': 'success',
            'insights': insights,
            'analysis_categories': {
                'descriptive': [i for i in insights if i.get('category') == 'descriptive'],
                'diagnostic': [i for i in insights if i.get('category') == 'diagnostic'],
                'prescriptive': [i for i in insights if i.get('category') == 'prescriptive'],
                'predictive': [i for i in insights if i.get('category') == 'predictive']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_insights: {str(e)}")
        return jsonify({'error': f'Insights generation failed: {str(e)}'}), 500

@app.route('/filter', methods=['POST'])
def apply_filters():
    """Apply filters to the current dataset"""
    global current_dataset
    
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded. Please analyze a dataset first.'}), 400
        
        # Get filter parameters
        data = request.get_json()
        filters = data.get('filters', {})
        
        # Initialize filter engine
        filter_engine = FilterEngine()
        
        # Apply filters
        filtered_df = filter_engine.apply_filters(current_dataset, filters)
        
        # Generate summary of filtered data
        filter_summary = filter_engine.get_filter_summary(current_dataset, filtered_df, filters)
        
        return jsonify({
            'status': 'success',
            'message': 'Filters applied successfully',
            'original_shape': current_dataset.shape,
            'filtered_shape': filtered_df.shape,
            'filter_summary': filter_summary,
            'sample_data': filtered_df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Error in apply_filters: {str(e)}")
        return jsonify({'error': f'Filter application failed: {str(e)}'}), 500

@app.route('/export', methods=['GET'])
def export_analysis():
    """Export analysis results and visualizations"""
    global current_analysis, current_visualizations
    
    try:
        if current_analysis is None:
            return jsonify({'error': 'No analysis available. Please analyze a dataset first.'}), 400
        
        # Create export data
        export_data = {
            'analysis': current_analysis,
            'visualizations': current_visualizations,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON and create file
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Create a BytesIO object
        output = BytesIO()
        output.write(json_data.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'analysis_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
    except Exception as e:
        logger.error(f"Error in export_analysis: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/available-filters', methods=['GET'])
def get_available_filters():
    """Get available filter options for the current dataset"""
    global current_dataset
    
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded. Please analyze a dataset first.'}), 400
        
        # Initialize filter engine
        filter_engine = FilterEngine()
        
        # Get available filters
        available_filters = filter_engine.get_available_filters(current_dataset)
        
        return jsonify({
            'status': 'success',
            'available_filters': available_filters
        })
        
    except Exception as e:
        logger.error(f"Error in get_available_filters: {str(e)}")
        return jsonify({'error': f'Failed to get available filters: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Starting AI Data Analysis Pipeline on {HOST}:{PORT}")
    logger.info(f"OpenAI API Key configured: {'Yes' if openai.api_key else 'No'}")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    ) 