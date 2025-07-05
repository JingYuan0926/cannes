# AI-Powered EDA Analysis System

An intelligent Exploratory Data Analysis (EDA) system that leverages OpenAI to generate automated insights, visualizations, and strategic recommendations from your datasets.

## 🚀 Features

### Core Capabilities
- **AI-Powered Analysis**: Uses OpenAI GPT to generate intelligent insights and recommendations
- **Comprehensive Data Analysis**: Descriptive, diagnostic, predictive, and prescriptive analytics
- **Interactive Visualizations**: Multiple chart types with Plotly integration
- **Data Quality Assessment**: Automated detection of missing values, outliers, and inconsistencies
- **Business Intelligence**: Strategic recommendations and actionable insights
- **Export Functionality**: Export results in PDF, JSON, and CSV formats

### Analysis Types
1. **Descriptive Analytics**: What happened in the data?
2. **Diagnostic Analytics**: Why did it happen?
3. **Predictive Analytics**: What might happen next?
4. **Prescriptive Analytics**: What should be done?

### Visualization Types
- Bar Charts
- Line Charts
- Scatter Plots
- Pie Charts
- Histograms
- Box Plots
- Heatmaps
- Area Charts
- Violin Plots
- Radar Charts

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (optional but recommended for AI insights)

### Installation

1. **Clone the repository and navigate to the analysis folder**:
   ```bash
   cd analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the analysis directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_ENV=development
   FLASK_DEBUG=True
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and go to `http://localhost:5000`

## 📊 Usage Guide

### Web Interface

1. **Upload Dataset**: 
   - Drag and drop a CSV file or click to browse
   - Maximum file size: 10MB
   - Supported format: CSV

2. **Set Analysis Parameters**:
   - **Prompt**: Optional custom prompt for specific analysis focus
   - **Analysis Type**: Choose from Descriptive, Diagnostic, Predictive, or All Types

3. **Review Results**:
   - **Executive Summary**: High-level overview of your dataset
   - **Key Metrics**: Important statistical measures
   - **Visualizations**: Interactive charts and graphs
   - **AI Insights**: Intelligent analysis from OpenAI
   - **Recommendations**: Strategic actions based on findings

4. **Export Results**:
   - PDF: Formatted report
   - JSON: Raw analysis data
   - CSV: Processed dataset

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Analyze Dataset
```bash
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: CSV file
- prompt: Optional analysis prompt
- analysis_type: descriptive|diagnostic|predictive|all
```

#### Generate Visualizations
```bash
POST /visualize
Content-Type: application/json

Body:
{
    "data": [...],
    "chart_type": "bar|line|scatter|pie|histogram|box|heatmap",
    "config": {...}
}
```

#### Get Business Insights
```bash
POST /insights
Content-Type: application/json

Body:
{
    "data": [...],
    "prompt": "Optional custom prompt"
}
```

#### Export Results
```bash
GET /export/{format}
Formats: pdf|json|csv
```

## 🏗️ Architecture

### Core Components

1. **Flask Application** (`app.py`):
   - Main web server and API endpoints
   - File upload handling
   - Request routing and response formatting

2. **Data Analyzer** (`utils/data_analyzer.py`):
   - Statistical analysis and data quality assessment
   - Pattern detection and outlier identification
   - Time series analysis

3. **Visualization Engine** (`utils/visualization_engine.py`):
   - Chart generation with Plotly
   - Automatic chart type selection
   - Interactive visualization creation

4. **Business Intelligence Engine** (`utils/business_intelligence.py`):
   - OpenAI integration for AI insights
   - Strategic recommendation generation
   - Comprehensive report creation

### Data Flow

```
CSV Upload → Data Validation → Analysis → Visualization → AI Insights → Results Display
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for AI insights | None |
| `FLASK_ENV` | Flask environment | production |
| `FLASK_DEBUG` | Enable debug mode | False |
| `MAX_CONTENT_LENGTH` | Maximum file upload size | 10MB |

### Customization

#### Adding New Chart Types
1. Add chart method to `VisualizationEngine` class
2. Update `CHART_TYPES` dictionary
3. Implement chart-specific logic

#### Extending Analysis Types
1. Add new analysis method to `DataAnalyzer` class
2. Update `comprehensive_analysis` method
3. Add corresponding UI elements

## 📈 Example Use Cases

### Sales Analysis
```python
# Upload sales data CSV
# Prompt: "Analyze sales trends and identify top-performing products"
# Analysis Type: All Types
```

### Customer Segmentation
```python
# Upload customer data CSV
# Prompt: "Segment customers based on purchase behavior and demographics"
# Analysis Type: Descriptive + Diagnostic
```

### Financial Performance
```python
# Upload financial data CSV
# Prompt: "Evaluate financial performance and identify cost optimization opportunities"
# Analysis Type: All Types
```

## 🚨 Troubleshooting

### Common Issues

1. **File Upload Errors**:
   - Check file size (max 10MB)
   - Ensure CSV format
   - Verify file permissions

2. **Analysis Failures**:
   - Check data quality (missing values, data types)
   - Verify OpenAI API key
   - Review error logs

3. **Visualization Issues**:
   - Ensure sufficient data points
   - Check column data types
   - Verify chart configuration

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and parameters |
| 413 | File Too Large | Reduce file size or increase limit |
| 500 | Internal Server Error | Check logs and configuration |

## 🔒 Security Considerations

- File upload validation and sanitization
- API key protection with environment variables
- Input validation and sanitization
- CORS configuration for cross-origin requests
- Rate limiting for API endpoints

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Create an issue in the repository
- Contact the development team

## 🔮 Future Enhancements

- Support for additional file formats (Excel, JSON, Parquet)
- Real-time data streaming analysis
- Advanced machine learning models
- Collaborative analysis features
- Dashboard customization options
- Scheduled analysis reports 