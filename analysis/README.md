# AI Data Analysis Pipeline

A comprehensive AI-powered data analysis and business intelligence system that automatically analyzes datasets, generates insights, and provides actionable recommendations using OpenAI's GPT models.

## 🚀 Features

### Core Analytics
- **AI-Powered Analysis**: Leverages OpenAI GPT models to understand data patterns and generate human-readable insights
- **Comprehensive Data Analysis**: Statistical analysis, correlation detection, trend identification, and outlier detection
- **Business Intelligence**: Strategic recommendations, performance metrics, and opportunity identification
- **Advanced Visualizations**: Interactive charts, dashboards, and customizable visualizations using Plotly
- **Smart Filtering**: Intelligent data filtering with natural language query support

### Advanced Capabilities
- **Trend Analysis**: Time series analysis, seasonality detection, and forecasting
- **Risk Assessment**: Automated risk identification and mitigation strategies
- **Performance Metrics**: KPI calculation and business health scoring
- **Data Quality Assessment**: Completeness, consistency, and accuracy evaluation
- **Custom Analytics**: Tailored analysis based on user prompts and business context

## 📊 Supported Data Formats

- CSV files
- Excel files (xlsx, xls)
- JSON files
- Parquet files (with configuration)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configurations
   ```

5. **Set up OpenAI API key**
   ```bash
   # Add your OpenAI API key to .env file
   OPENAI_API_KEY=your_api_key_here
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

The service will be available at `http://localhost:3032`

## 📡 API Endpoints

### Health Check
```http
GET /
```
Returns service status and health information.

### Dataset Analysis
```http
POST /analyze
```
Analyzes an uploaded dataset and returns comprehensive insights.

**Parameters:**
- `file`: Dataset file (CSV, Excel, JSON)
- `prompt` (optional): Business question or analysis focus

**Response:**
```json
{
  "analysis_id": "unique_id",
  "dataset_info": {
    "rows": 1000,
    "columns": 15,
    "data_types": {...}
  },
  "ai_recommendations": [...],
  "statistical_summary": {...},
  "correlations": {...},
  "insights": {...}
}
```

### Generate Visualizations
```http
POST /visualize
```
Creates interactive visualizations based on analysis results.

**Parameters:**
- `analysis_id`: ID from previous analysis
- `chart_type`: Type of chart (bar, line, scatter, etc.)
- `x_axis`: Column for x-axis
- `y_axis`: Column for y-axis
- `config`: Additional chart configuration

**Response:**
```json
{
  "chart_html": "<html>...</html>",
  "chart_json": {...},
  "insights": [...],
  "business_context": "..."
}
```

### Business Intelligence Insights
```http
POST /insights
```
Generates comprehensive business intelligence insights.

**Parameters:**
- `analysis_id`: ID from previous analysis
- `focus_area`: Business focus (growth, performance, risk, etc.)
- `user_prompt`: Specific business question

**Response:**
```json
{
  "executive_summary": {...},
  "key_findings": [...],
  "strategic_recommendations": [...],
  "performance_metrics": {...},
  "risk_assessment": {...},
  "opportunities": [...]
}
```

### Apply Data Filters
```http
POST /filter
```
Applies intelligent filters to the dataset.

**Parameters:**
- `analysis_id`: ID from previous analysis
- `filters`: Array of filter configurations
- `natural_language_query`: Natural language filter description

**Response:**
```json
{
  "filtered_data_id": "new_id",
  "original_rows": 1000,
  "filtered_rows": 750,
  "applied_filters": [...],
  "summary": {...}
}
```

### Export Results
```http
GET /export/{analysis_id}
```
Exports analysis results in various formats.

**Query Parameters:**
- `format`: Export format (csv, xlsx, json, pdf)
- `include_charts`: Include visualizations (true/false)

### Available Filters
```http
GET /available-filters/{analysis_id}
```
Returns available filter options for the dataset.

## 💡 Usage Examples

### Python Client Example
```python
import requests
import json

# Upload and analyze dataset
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:3032/analyze',
        files={'file': f},
        data={'prompt': 'I want to see my company growth over the last 5 years'}
    )

analysis_result = response.json()
analysis_id = analysis_result['analysis_id']

# Generate visualization
viz_response = requests.post(
    'http://localhost:3032/visualize',
    json={
        'analysis_id': analysis_id,
        'chart_type': 'line',
        'x_axis': 'date',
        'y_axis': 'revenue',
        'config': {'title': 'Revenue Growth Over Time'}
    }
)

# Get business insights
insights_response = requests.post(
    'http://localhost:3032/insights',
    json={
        'analysis_id': analysis_id,
        'focus_area': 'growth',
        'user_prompt': 'What are the key growth drivers?'
    }
)
```

### cURL Examples
```bash
# Analyze dataset
curl -X POST http://localhost:3032/analyze \
  -F "file=@data.csv" \
  -F "prompt=Analyze sales performance"

# Create visualization
curl -X POST http://localhost:3032/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "abc123",
    "chart_type": "bar",
    "x_axis": "category",
    "y_axis": "sales"
  }'

# Apply filters
curl -X POST http://localhost:3032/filter \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "abc123",
    "natural_language_query": "Show me sales above 1000 for the last quarter"
  }'
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | gpt-3.5-turbo |
| `PORT` | Server port | 3032 |
| `MAX_FILE_SIZE` | Maximum upload size | 100MB |
| `ENABLE_AI_INSIGHTS` | Enable AI-powered insights | True |
| `LOG_LEVEL` | Logging level | INFO |

### Analysis Configuration
- **MAX_ROWS_FOR_ANALYSIS**: Maximum rows to analyze (default: 100,000)
- **ENABLE_CACHING**: Enable result caching (default: True)
- **INSIGHT_CONFIDENCE_THRESHOLD**: Minimum confidence for insights (default: 0.6)

### Visualization Configuration
- **CHART_THEME**: Default chart theme (default: plotly_white)
- **CHART_WIDTH/HEIGHT**: Default chart dimensions (800x600)
- **ENABLE_INTERACTIVE_CHARTS**: Enable interactive features (default: True)

## 🎯 Business Intelligence Features

### Executive Dashboard
- Dataset overview and health metrics
- Key performance indicators
- Critical alerts and opportunities
- Business health scoring

### Strategic Analysis
- Growth pattern identification
- Performance optimization recommendations
- Risk assessment and mitigation
- Market opportunity analysis

### Operational Insights
- Process improvement suggestions
- Efficiency optimization
- Resource allocation recommendations
- Quality improvement strategies

## 🧠 AI-Powered Features

### Natural Language Processing
- Understands business questions in plain English
- Contextual analysis based on user prompts
- Industry-specific insights and recommendations
- Automated report generation

### Intelligent Recommendations
- Data-driven strategic suggestions
- Performance improvement actions
- Risk mitigation strategies
- Growth opportunity identification

### Adaptive Analysis
- Automatically selects appropriate analysis techniques
- Customizes insights based on data characteristics
- Provides context-aware recommendations
- Learns from user feedback

## 📈 Visualization Types

### Standard Charts
- Bar charts (grouped, stacked, horizontal)
- Line charts (single, multi-series, area)
- Scatter plots (with trend lines)
- Pie charts and donut charts
- Histograms and box plots

### Advanced Visualizations
- Correlation heatmaps
- Treemaps and sunburst charts
- Funnel and waterfall charts
- Gauge charts and indicators
- Time series decomposition

### Interactive Features
- Zoom and pan capabilities
- Hover information and tooltips
- Dynamic filtering and selection
- Export to various formats
- Responsive design

## 🔍 Data Quality Assessment

### Completeness Analysis
- Missing value detection and patterns
- Data coverage assessment
- Completeness scoring and recommendations

### Consistency Evaluation
- Data type consistency
- Format standardization
- Duplicate detection and handling

### Accuracy Indicators
- Outlier detection and analysis
- Data validation rules
- Quality scoring and improvement suggestions

## 🚨 Error Handling

The service includes comprehensive error handling for:
- Invalid file formats or corrupted data
- Large dataset processing limits
- API rate limiting and timeouts
- Memory and resource constraints
- Network connectivity issues

## 📊 Performance Optimization

### Caching Strategy
- Analysis result caching
- Visualization caching
- Smart cache invalidation

### Resource Management
- Memory-efficient data processing
- Chunked analysis for large datasets
- Parallel processing capabilities
- Resource monitoring and limits

## 🔒 Security Features

- File type validation and sanitization
- Size limits and resource constraints
- API key protection and validation
- CORS configuration for web access
- Input validation and sanitization

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=.
```

## 📝 Development

### Code Structure
```
analysis/
├── app.py                 # Main Flask application
├── utils/
│   ├── __init__.py
│   ├── data_analyzer.py   # Core analysis engine
│   ├── visualization_engine.py  # Chart generation
│   ├── filter_engine.py   # Data filtering
│   └── business_intelligence.py  # BI insights
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

### Adding New Features
1. Create feature branch
2. Implement functionality in appropriate module
3. Add tests for new features
4. Update API documentation
5. Submit pull request

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Large Dataset Processing**
- Increase `MAX_ROWS_FOR_ANALYSIS` in configuration
- Enable chunked processing for very large files
- Consider data sampling for initial analysis

**OpenAI API Limits**
- Monitor API usage and rate limits
- Implement retry logic with exponential backoff
- Consider using different models for different tasks

**Memory Issues**
- Adjust `MEMORY_LIMIT` configuration
- Enable multiprocessing for better resource utilization
- Use data sampling for memory-intensive operations

**Visualization Performance**
- Reduce data points for large datasets
- Use aggregation for better performance
- Enable chart caching for repeated requests

### Getting Help
- Check the logs for detailed error information
- Review configuration settings
- Consult the API documentation
- Submit issues on GitHub

## 🗺️ Roadmap

### Upcoming Features
- Real-time data streaming support
- Advanced machine learning models
- Custom dashboard builder
- Multi-language support
- Enhanced forecasting capabilities
- Integration with popular BI tools
- Automated report scheduling
- Advanced statistical tests
- Custom metric definitions
- Team collaboration features

### Performance Improvements
- Distributed processing support
- Advanced caching strategies
- Database integration options
- Streaming data processing
- GPU acceleration for ML tasks 