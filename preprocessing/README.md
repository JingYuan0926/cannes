# AI Data Preprocessing Agent

A comprehensive AI-powered data preprocessing service that automatically analyzes datasets and applies intelligent preprocessing techniques using OpenAI's GPT models. This service provides advanced data standardization, normalization, feature engineering, and preprocessing capabilities.

## Features

- **AI-Powered Analysis**: Uses OpenAI GPT-4 to analyze datasets and recommend preprocessing techniques
- **Comprehensive Preprocessing**: 10+ preprocessing techniques including scaling, encoding, feature engineering, and more
- **Intelligent Recommendations**: Automatically suggests the best preprocessing pipeline based on data characteristics
- **REST API**: Easy-to-use Flask-based REST API for integration with other applications
- **Advanced Techniques**: Supports dimensionality reduction, outlier detection, text processing, and datetime feature engineering
- **Flexible Configuration**: Supports both AI-driven and custom preprocessing pipelines

## Available Preprocessing Techniques

### 1. Numerical Feature Standardization
- **Z-Score Normalization**: Standard normalization (mean=0, std=1)
- **Min-Max Scaling**: Scale features to [0,1] range
- **Robust Scaling**: Scale using median and IQR (robust to outliers)
- **Quantile Transformation**: Transform to uniform or normal distribution

### 2. Categorical Feature Encoding
- **One-Hot Encoding**: Create binary columns for each category
- **Label Encoding**: Assign integer labels to categories
- **Frequency Encoding**: Replace categories with frequency counts
- **Target Encoding**: Replace categories with target statistics
- **Binary Encoding**: Encode using binary representation

### 3. Text Feature Processing
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
- **Count Vectorization**: Count-based text vectorization
- **Text Preprocessing**: Basic text cleaning and normalization

### 4. Datetime Feature Engineering
- **Component Extraction**: Year, month, day, hour, weekday, quarter
- **Derived Features**: Weekend indicator, days since epoch
- **Temporal Patterns**: Seasonal and cyclical feature extraction

### 5. Advanced Techniques
- **Polynomial Features**: Create polynomial and interaction features
- **Feature Selection**: Variance threshold, correlation analysis
- **Outlier Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM
- **Dimensionality Reduction**: PCA, Factor Analysis, t-SNE
- **Imbalanced Data Handling**: SMOTE, ADASYN, over/under sampling

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd preprocessing
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
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. **Run the application**:
```bash
python app.py
```

The service will start on `http://localhost:3031`

## API Endpoints

### Health Check
```http
GET /
```
Returns service status and health information.

### Analyze Dataset
```http
POST /analyze
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: CSV file to analyze

**Response:**
```json
{
  "status": "success",
  "message": "Dataset analyzed successfully",
  "analysis": {
    "basic_info": {...},
    "numerical_analysis": {...},
    "categorical_analysis": {...},
    "distribution_analysis": {...},
    "scaling_requirements": {...},
    "encoding_requirements": {...},
    "feature_engineering": {...},
    "preprocessing_recommendations": {...}
  }
}
```

### AI-Powered Preprocessing
```http
POST /preprocess
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: CSV file to preprocess

**Response:**
```json
{
  "status": "success",
  "message": "Data preprocessed successfully",
  "original_shape": [100, 10],
  "processed_shape": [100, 15],
  "recommendations": {...},
  "execution_log": [...],
  "output_file": "data/preprocessed.csv"
}
```

### Custom Preprocessing
```http
POST /custom-preprocess
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: CSV file to preprocess
- `config`: JSON configuration for preprocessing pipeline

**Example config:**
```json
{
  "preprocessing_pipeline": [
    {
      "technique": "standardize_numerical_features",
      "parameters": {"method": "z_score"},
      "columns": ["age", "income"],
      "priority": 8
    },
    {
      "technique": "normalize_categorical_features",
      "parameters": {"method": "one_hot_encoding"},
      "columns": ["category"],
      "priority": 7
    }
  ]
}
```

### Download Preprocessed Data
```http
GET /download/preprocessed
```
Downloads the preprocessed CSV file.

### Get Available Techniques
```http
GET /techniques
```
Returns all available preprocessing techniques with descriptions and parameters.

## Usage Examples

### Python Client Example
```python
import requests
import pandas as pd

# Upload and analyze dataset
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:3031/analyze', files={'file': f})
    analysis = response.json()

print("Dataset Analysis:")
print(f"Shape: {analysis['analysis']['basic_info']['shape']}")
print(f"Columns: {analysis['analysis']['basic_info']['columns']}")

# AI-powered preprocessing
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:3031/preprocess', files={'file': f})
    result = response.json()

print(f"Original shape: {result['original_shape']}")
print(f"Processed shape: {result['processed_shape']}")

# Download preprocessed data
response = requests.get('http://localhost:3031/download/preprocessed')
with open('preprocessed_data.csv', 'wb') as f:
    f.write(response.content)
```

### cURL Examples
```bash
# Analyze dataset
curl -X POST -F "file=@data.csv" http://localhost:3031/analyze

# Preprocess dataset
curl -X POST -F "file=@data.csv" http://localhost:3031/preprocess

# Get available techniques
curl http://localhost:3031/techniques

# Download preprocessed data
curl -O http://localhost:3031/download/preprocessed
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required for AI recommendations)
- `PORT`: Server port (default: 3031)
- `HOST`: Server host (default: 0.0.0.0)
- `FLASK_ENV`: Flask environment (development/production)
- `FLASK_DEBUG`: Enable debug mode (True/False)

### Preprocessing Configuration
The service supports both AI-driven and manual configuration:

**AI-Driven**: The service analyzes your data and uses OpenAI to recommend the best preprocessing pipeline.

**Manual Configuration**: You can specify custom preprocessing pipelines using the `/custom-preprocess` endpoint.

## Advanced Features

### Feature Engineering
- **Polynomial Features**: Create polynomial combinations of numerical features
- **Interaction Features**: Generate multiplicative, additive, and ratio features
- **Datetime Engineering**: Extract temporal patterns and seasonal features
- **Text Processing**: Convert text to numerical features using TF-IDF or count vectorization

### Data Quality Enhancement
- **Outlier Detection**: Multiple algorithms for identifying and handling outliers
- **Feature Selection**: Remove low-variance and highly correlated features
- **Dimensionality Reduction**: PCA, Factor Analysis, and t-SNE for dimension reduction
- **Imbalanced Data**: SMOTE and other techniques for handling class imbalance

### Intelligent Analysis
- **Distribution Analysis**: Identify data distributions and recommend transformations
- **Scaling Requirements**: Analyze feature scales and recommend appropriate scalers
- **Encoding Requirements**: Analyze categorical variables and recommend encoding methods
- **Correlation Analysis**: Identify multicollinearity and feature relationships

## Error Handling

The service includes comprehensive error handling:
- Invalid file formats
- Missing required parameters
- OpenAI API errors (falls back to rule-based recommendations)
- Data processing errors
- File size limitations

## Monitoring and Logging

- Comprehensive logging of all preprocessing operations
- Execution logs for each preprocessing step
- Performance metrics and timing information
- Error tracking and debugging information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the error logs in the console
2. Verify your OpenAI API key is set correctly
3. Ensure your data file is in CSV format
4. Check that all required dependencies are installed

## Roadmap

- [ ] Support for more file formats (Excel, JSON, Parquet)
- [ ] Advanced feature engineering techniques
- [ ] Model-specific preprocessing pipelines
- [ ] Automated hyperparameter tuning for preprocessing
- [ ] Integration with popular ML frameworks
- [ ] Real-time preprocessing capabilities
- [ ] Batch processing for large datasets 