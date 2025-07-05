"""
Configuration file for the AI-Powered EDA Analysis System.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', '2000'))
    OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE', '0.7'))
    
    # Analysis Configuration
    DEFAULT_ANALYSIS_TYPE = 'all'
    MAX_VISUALIZATION_COUNT = 10
    MAX_INSIGHT_COUNT = 20
    
    # Data Quality Thresholds
    MISSING_DATA_THRESHOLD = 0.1  # 10% missing data threshold
    OUTLIER_THRESHOLD = 0.05  # 5% outlier threshold
    CORRELATION_THRESHOLD = 0.7  # Strong correlation threshold
    
    # Chart Configuration
    CHART_HEIGHT = 400
    CHART_WIDTH = 600
    CHART_COLORS = [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
        '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3'
    ]
    
    # Export Configuration
    EXPORT_FOLDER = 'exports'
    PDF_TEMPLATE = 'report_template.html'
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'analysis.log'
    
    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = '100 per hour'
    
    # CORS Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8000']
    
    @staticmethod
    def init_app(app):
        """Initialize the Flask application with configuration."""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.EXPORT_FOLDER, exist_ok=True)
        
        # Set Flask configuration
        app.config.from_object(Config)
        
        # Configure logging
        import logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler()
            ]
        )

class DevelopmentConfig(Config):
    """Development configuration."""
    FLASK_ENV = 'development'
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    FLASK_ENV = 'production'
    FLASK_DEBUG = False
    LOG_LEVEL = 'INFO'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
    LOG_LEVEL = 'WARNING'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Analysis prompts for different use cases
ANALYSIS_PROMPTS = {
    'sales': """
    Analyze the sales data to identify:
    1. Top performing products and categories
    2. Sales trends over time
    3. Regional performance differences
    4. Customer segmentation insights
    5. Seasonal patterns and opportunities
    """,
    
    'customer': """
    Analyze customer data to understand:
    1. Customer demographics and behavior
    2. Purchase patterns and preferences
    3. Customer lifetime value segments
    4. Churn risk indicators
    5. Retention opportunities
    """,
    
    'financial': """
    Analyze financial data to evaluate:
    1. Revenue and profitability trends
    2. Cost structure analysis
    3. Budget vs actual performance
    4. Financial ratios and KPIs
    5. Investment opportunities
    """,
    
    'marketing': """
    Analyze marketing data to assess:
    1. Campaign performance and ROI
    2. Channel effectiveness
    3. Customer acquisition costs
    4. Conversion funnel analysis
    5. Attribution modeling insights
    """,
    
    'operations': """
    Analyze operational data to optimize:
    1. Process efficiency metrics
    2. Resource utilization patterns
    3. Quality control indicators
    4. Supply chain performance
    5. Operational bottlenecks
    """
}

# Chart type recommendations based on data characteristics
CHART_RECOMMENDATIONS = {
    'categorical_single': ['bar', 'pie'],
    'categorical_multiple': ['bar', 'heatmap'],
    'numerical_single': ['histogram', 'box'],
    'numerical_multiple': ['scatter', 'line', 'heatmap'],
    'time_series': ['line', 'area'],
    'correlation': ['scatter', 'heatmap'],
    'distribution': ['histogram', 'box', 'violin']
}

# Business metrics templates
BUSINESS_METRICS = {
    'revenue': ['Sales', 'Revenue', 'Income', 'Earnings'],
    'cost': ['Cost', 'Expense', 'Expenditure', 'Spending'],
    'quantity': ['Quantity', 'Volume', 'Count', 'Units'],
    'rate': ['Rate', 'Percentage', 'Ratio', 'Conversion'],
    'time': ['Date', 'Time', 'Timestamp', 'Period'],
    'geography': ['Region', 'Country', 'State', 'City', 'Location'],
    'customer': ['Customer', 'Client', 'User', 'Account'],
    'product': ['Product', 'Item', 'SKU', 'Category']
}

# Data quality checks
DATA_QUALITY_CHECKS = {
    'completeness': {
        'missing_values': 'Check for missing values in critical columns',
        'null_percentage': 'Calculate percentage of null values per column'
    },
    'consistency': {
        'data_types': 'Verify appropriate data types for each column',
        'format_consistency': 'Check date and number format consistency'
    },
    'accuracy': {
        'outliers': 'Identify statistical outliers in numerical data',
        'range_validation': 'Validate data within expected ranges'
    },
    'uniqueness': {
        'duplicates': 'Check for duplicate records',
        'key_uniqueness': 'Verify uniqueness of key columns'
    }
} 