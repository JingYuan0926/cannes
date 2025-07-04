# AI Data Cleaning Agent

An intelligent Flask-based data cleaning service that uses OpenAI's GPT models to analyze datasets and provide intelligent cleaning recommendations.

## Features

- **AI-Powered Analysis**: Uses OpenAI to analyze data quality issues and provide intelligent cleaning recommendations
- **Comprehensive Data Analysis**: Detects missing values, duplicates, outliers, data type issues, and more
- **Multiple Cleaning Strategies**: Supports various cleaning methods including missing value handling, outlier detection, text standardization, and data type conversion
- **RESTful API**: Easy-to-use REST endpoints for data analysis and cleaning
- **Flexible Configuration**: Customizable cleaning parameters and strategies
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Installation

1. Clone the repository and navigate to the ETL directory:
```bash
cd etl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export FLASK_ENV="development"  # or "production"
```

4. Run the application:
```bash
python app.py
```

The service will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/` - Check service status

### Data Analysis
- **POST** `/analyze` - Analyze uploaded CSV data
  - Upload a CSV file to get comprehensive data quality analysis
  - Returns: Dataset overview, missing data analysis, duplicate detection, outlier analysis, and more

### Data Cleaning
- **POST** `/clean` - Clean data using AI recommendations
  - Upload a CSV file to get AI-powered cleaning recommendations and execute them
  - Returns: Cleaned dataset analysis and download link

### Download Cleaned Data
- **GET** `/download/cleaned` - Download the cleaned CSV file

### Available Strategies
- **GET** `/strategies` - Get all available cleaning strategies and their parameters

### Custom Cleaning
- **POST** `/custom-clean` - Clean data with custom parameters
  - Specify your own cleaning strategies and parameters
  - Body: JSON with cleaning configuration

## Usage Examples

### Basic Analysis
```bash
curl -X POST -F "file=@your_data.csv" http://localhost:5000/analyze
```

### AI-Powered Cleaning
```bash
curl -X POST -F "file=@your_data.csv" http://localhost:5000/clean
```

### Custom Cleaning
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": [
      {
        "name": "handle_missing_values",
        "parameters": {"strategy": "fill_median"}
      },
      {
        "name": "remove_duplicates",
        "parameters": {"keep": "first"}
      }
    ]
  }' \
  -F "file=@your_data.csv" \
  http://localhost:5000/custom-clean
```

## Available Cleaning Strategies

### 1. Handle Missing Values
- **Strategy**: `handle_missing_values`
- **Options**: drop, fill_mean, fill_median, fill_mode, forward_fill, backward_fill, interpolate, fill_value, fill_zero
- **Use Cases**: High missing data percentage, systematic missing patterns

### 2. Remove Duplicates
- **Strategy**: `remove_duplicates`
- **Options**: Keep first, last, or remove all duplicates
- **Use Cases**: Exact duplicate rows, data entry errors

### 3. Handle Outliers
- **Strategy**: `handle_outliers`
- **Methods**: IQR, z-score, isolation forest, percentile, capping
- **Use Cases**: Statistical outliers, measurement errors

### 4. Standardize Text
- **Strategy**: `standardize_text`
- **Operations**: lowercase, uppercase, strip, remove special characters, normalize whitespace
- **Use Cases**: Inconsistent text formatting, mixed case data

### 5. Convert Data Types
- **Strategy**: `convert_data_types`
- **Options**: datetime, numeric, categorical conversions
- **Use Cases**: Incorrect data types after import

### 6. Handle Categorical Data
- **Strategy**: `handle_categorical_data`
- **Encoding**: label, one-hot, frequency, target encoding
- **Use Cases**: Machine learning preprocessing

### 7. Normalize Numerical Data
- **Strategy**: `normalize_numerical_data`
- **Methods**: standard, min-max, robust scaling, log, sqrt transformations
- **Use Cases**: Different scales between features

### 8. Validate Data Consistency
- **Strategy**: `validate_data_consistency`
- **Rules**: Range checks, format validation, business rules
- **Use Cases**: Data quality validation

### 9. Remove Constant Columns
- **Strategy**: `remove_constant_columns`
- **Options**: Configurable threshold for near-constant detection
- **Use Cases**: Uninformative columns, feature selection

### 10. Handle Date Columns
- **Strategy**: `handle_date_columns`
- **Features**: Date parsing, feature extraction (year, month, day, weekend)
- **Use Cases**: Date string parsing, time series analysis

### 11. Clean Column Names
- **Strategy**: `clean_column_names`
- **Operations**: Standardize naming conventions, remove special characters
- **Use Cases**: Inconsistent column naming

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `FLASK_ENV`: Flask environment (development/production)

### Logging
The application uses structured logging with different levels:
- INFO: General operation information
- WARNING: Data quality issues and recommendations
- ERROR: Processing errors and failures
- DEBUG: Detailed debugging information

## Architecture

The system consists of several key components:

1. **Flask Application** (`app.py`): Main API server with endpoints
2. **AI Data Cleaning Agent**: Core intelligence using OpenAI for recommendations
3. **Data Analyzer** (`utils/data_analyzer.py`): Comprehensive data quality analysis
4. **Data Cleaner** (`utils/data_cleaner.py`): Implementation of cleaning strategies
5. **Cleaning Strategies** (`utils/cleaning_strategies.py`): Strategy registry and configuration

## Error Handling

The application includes comprehensive error handling:
- File upload validation
- Data format validation
- OpenAI API error handling
- Graceful fallback to rule-based recommendations
- Detailed error logging and user feedback

## Security Considerations

- API keys are loaded from environment variables
- File uploads are validated for type and size
- Temporary files are cleaned up after processing
- Input validation for all API endpoints

## Performance

- Efficient pandas operations for data processing
- Streaming file uploads for large datasets
- Optimized outlier detection algorithms
- Caching of analysis results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the logs for detailed error information
2. Verify your OpenAI API key is correctly set
3. Ensure all dependencies are properly installed
4. Review the API documentation for correct usage

## Changelog

### Version 1.0.0
- Initial release with AI-powered data cleaning
- Support for 11 different cleaning strategies
- RESTful API with comprehensive endpoints
- Detailed data analysis and reporting
- OpenAI integration for intelligent recommendations 