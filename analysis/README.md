# AI-Powered Data Analysis System

A comprehensive AI-powered data analysis system that utilizes OpenAI's language models to intelligently analyze datasets and provide actionable insights through advanced analytics algorithms.

## 🚀 Features

### Analytics Categories

1. **Descriptive Analytics** - "What happened?"
   - K-Means Clustering
   - DBSCAN Clustering
   - Principal Component Analysis (PCA)
   - t-SNE Analysis

2. **Predictive Analytics** - "What will happen?"
   - Linear Regression
   - Ridge Regression
   - Decision Tree Regression
   - Logistic Regression
   - Random Forest Classification
   - Support Vector Machine (SVM)

3. **Prescriptive Analytics** - "What should we do?"
   - Linear Programming
   - Genetic Algorithm
   - Collaborative Filtering
   - Content-Based Filtering
   - Portfolio Optimization
   - Resource Allocation

4. **Diagnostic Analytics** - "Why did it happen?"
   - Decision Tree Analysis
   - Feature Importance Analysis
   - Causal Inference
   - Correlation Analysis
   - Anomaly Detection
   - Root Cause Analysis

### Key Capabilities

- **AI-Powered Analysis Strategy**: OpenAI determines the best analysis approach based on dataset characteristics and user goals
- **Intelligent Insights**: AI-generated insights, justifications, conclusions, and recommendations
- **Interactive Visualizations**: High-quality charts and graphs using Plotly
- **Multiple File Formats**: Support for CSV, Excel, JSON, and Parquet files
- **RESTful API**: Easy integration with other systems
- **Comprehensive Results**: JSON output with detailed analysis results

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- 16MB+ available memory for file uploads

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the analysis directory:
   ```bash
   cd analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Create required directories**:
   ```bash
   mkdir -p uploads results
   ```

## 🚀 Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:3040` by default.

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Service Status
```bash
GET /
```

#### 3. Analyze Dataset
```bash
POST /analyze
```

**File Upload Example:**
```bash
curl -X POST -F "file=@your_dataset.csv" -F "goal=Analyze sales performance and identify trends" http://localhost:3040/analyze
```

**JSON Data Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "data": [...],
  "goal": "Understand customer behavior patterns"
}' http://localhost:3040/analyze
```

#### 4. Get Analysis Results
```bash
GET /results/<analysis_id>
```

#### 5. List All Analyses
```bash
GET /list-analyses
```

### Example Usage

1. **Upload a dataset and get AI-powered analysis:**
   ```bash
   curl -X POST -F "file=@sales_data.csv" -F "goal=Identify factors affecting sales performance" http://localhost:3040/analyze
   ```

2. **The system will:**
   - Analyze your dataset characteristics
   - Use AI to determine the best analysis approach
   - Execute the recommended algorithms
   - Generate insights and visualizations
   - Return comprehensive results in JSON format

3. **Sample response structure:**
   ```json
   {
     "analysis_id": "uuid-here",
     "success": true,
     "ai_strategy": {
       "recommended_category": "diagnostic",
       "recommended_algorithm": "feature_importance_analysis",
       "reasoning": "Based on the dataset characteristics..."
     },
     "analysis_results": {
       "algorithm": "Feature Importance Analysis",
       "results": {...},
       "insights": [...],
       "graphs": [...]
     },
     "ai_insights": {
       "insights": [...],
       "justifications": [...],
       "conclusions": [...],
       "recommendations": [...]
     }
   }
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `DEBUG` | Debug mode | `True` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `3040` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model | `gpt-4` |
| `OPENAI_TEMPERATURE` | AI creativity level | `0.7` |

### Supported File Formats

- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **JSON** (`.json`)
- **Parquet** (`.parquet`)

### File Size Limits

- Maximum file size: 16MB
- Maximum analysis results stored: 100

## 🏗️ Architecture

```
analysis/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # This file
└── utils/                          # Analysis modules
    ├── analysis_orchestrator.py    # Main orchestrator with AI integration
    ├── descriptive_analytics.py    # Clustering and dimensionality reduction
    ├── predictive_analytics.py     # Regression and classification
    ├── prescriptive_analytics.py   # Optimization and recommendations
    └── diagnostic_analytics.py     # Feature importance and causal analysis
```

## 📊 Analytics Algorithms

### Descriptive Analytics
- **K-Means Clustering**: Groups similar data points
- **DBSCAN**: Density-based clustering for anomaly detection
- **PCA**: Dimensionality reduction and variance analysis
- **t-SNE**: Non-linear dimensionality reduction for visualization

### Predictive Analytics
- **Linear/Ridge Regression**: Predicts continuous values
- **Decision Tree Regression**: Non-linear regression with interpretable rules
- **Logistic Regression**: Binary classification
- **Random Forest**: Ensemble classification method
- **SVM**: Support vector machine for classification

### Prescriptive Analytics
- **Linear Programming**: Optimization with linear constraints
- **Genetic Algorithm**: Evolutionary optimization
- **Collaborative Filtering**: Recommendation based on user behavior
- **Content-Based Filtering**: Recommendation based on item features
- **Portfolio Optimization**: Financial portfolio optimization
- **Resource Allocation**: Optimal resource distribution

### Diagnostic Analytics
- **Decision Tree Analysis**: Interpretable if-then rules
- **Feature Importance**: Identifies most influential variables
- **Causal Inference**: Discovers cause-effect relationships
- **Correlation Analysis**: Finds variable relationships
- **Anomaly Detection**: Identifies outliers and anomalies
- **Root Cause Analysis**: Determines factors driving outcomes

## 🎯 Use Cases

1. **Business Intelligence**: Analyze sales, customer, and operational data
2. **Financial Analysis**: Risk assessment, portfolio optimization
3. **Marketing Analytics**: Customer segmentation, campaign effectiveness
4. **Operations Research**: Resource allocation, process optimization
5. **Quality Control**: Anomaly detection, root cause analysis
6. **Research & Development**: Experimental data analysis, hypothesis testing

## 🔒 Security

- File upload validation
- Input sanitization
- Error handling and logging
- Environment variable protection
- API key security

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the key has sufficient credits

2. **File Upload Errors**
   - Verify file format is supported
   - Check file size is under 16MB
   - Ensure file is not corrupted

3. **Analysis Failures**
   - Check dataset has sufficient data (minimum 10 rows)
   - Verify dataset contains appropriate column types
   - Review logs for specific error messages

### Logs

Check `analysis.log` for detailed error information and debugging.

## 📈 Performance

- **Processing Time**: Varies by dataset size and algorithm complexity
- **Memory Usage**: Scales with dataset size
- **Concurrent Requests**: Supports multiple simultaneous analyses
- **Result Caching**: Analysis results are stored for future reference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Create an issue with detailed information

---

**Built with ❤️ using Python, Flask, scikit-learn, and OpenAI** 