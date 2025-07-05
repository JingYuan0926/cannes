# AI Data Analysis Pipeline - Testing Guide

This guide will teach you how to thoroughly test your AI Data Analysis Pipeline using multiple testing approaches.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Test Types](#test-types)
3. [Running Tests](#running-tests)
4. [Manual API Testing](#manual-api-testing)
5. [Performance Testing](#performance-testing)
6. [Test Data](#test-data)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Run All Tests at Once
```bash
cd analysis
python run_tests.py
```

### 2. Run Specific Test Types
```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only  
python run_tests.py --integration

# API tests only (requires Flask app running)
python run_tests.py --api

# Performance tests only
python run_tests.py --performance
```

### 3. Generate Test Report
```bash
python run_tests.py --report
```

## 🧪 Test Types

### 1. Unit Tests
**Location**: `tests/test_*.py`
**Purpose**: Test individual components in isolation

- **Data Analyzer Tests** (`test_data_analyzer.py`)
  - Basic statistics calculation
  - Data quality assessment
  - Outlier detection
  - Column analysis
  - Edge cases (empty data, NaN values)

- **Visualization Engine Tests** (`test_visualization_engine.py`)
  - Chart creation (bar, line, scatter, pie, etc.)
  - Auto chart selection
  - Dashboard creation
  - Error handling

- **API Endpoint Tests** (`test_api_endpoints.py`)
  - Flask route testing
  - File upload handling
  - JSON data processing
  - Error responses

### 2. Integration Tests
**Location**: `test_setup.py`
**Purpose**: Test component interactions

- Import verification
- Class instantiation
- End-to-end data flow
- Environment setup

### 3. API Tests
**Location**: `tests/test_manual_api.py`
**Purpose**: Test HTTP endpoints with real requests

- Health check endpoint
- Data upload and analysis
- Visualization generation
- Filtering operations
- Business insights
- Data export

### 4. Performance Tests
**Purpose**: Measure system performance with different data sizes

- Processing time analysis
- Memory usage monitoring
- Scalability assessment

## 🏃 Running Tests

### Method 1: Using the Test Runner (Recommended)

```bash
# Run all tests with verbose output
python run_tests.py --verbose

# Run specific tests
python run_tests.py --unit --verbose
python run_tests.py --integration
python run_tests.py --api
python run_tests.py --performance
```

### Method 2: Using Python unittest

```bash
# Run all unit tests
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_data_analyzer -v

# Run specific test method
python -m unittest tests.test_data_analyzer.TestDataAnalyzer.test_comprehensive_analysis -v
```

### Method 3: Using pytest (if installed)

```bash
# Install pytest first
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=utils --cov-report=html
```

## 🌐 Manual API Testing

### Step 1: Start the Flask Application
```bash
python app.py
```

### Step 2: Run API Tests
```bash
# In a new terminal
python tests/test_manual_api.py
```

### Step 3: Test Individual Endpoints

#### Health Check
```bash
curl http://localhost:3032/health
```

#### Upload and Analyze CSV
```bash
curl -X POST -F "file=@sample_data/sample_sales_data.csv" http://localhost:3032/analyze
```

#### Create Visualization
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"chart_type":"bar","x_axis":"category","y_axis":"sales","title":"Sales by Category"}' \
  http://localhost:3032/visualize
```

#### Apply Filters
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filters":[{"column":"sales","type":"greater_than","value":2000}]}' \
  http://localhost:3032/filter
```

#### Get Business Insights
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"What are the key trends in sales data?","focus_areas":["trends","performance"]}' \
  http://localhost:3032/insights
```

## ⚡ Performance Testing

### Automated Performance Tests
```bash
python run_tests.py --performance
```

### Manual Performance Testing

#### 1. Test with Different Data Sizes
```python
import pandas as pd
import numpy as np
import time
from utils.data_analyzer import DataAnalyzer

# Test with various data sizes
sizes = [100, 1000, 10000, 100000]
for size in sizes:
    # Create test data
    data = pd.DataFrame({
        'values': np.random.randn(size),
        'categories': np.random.choice(['A', 'B', 'C'], size)
    })
    
    # Time the analysis
    analyzer = DataAnalyzer()
    start_time = time.time()
    result = analyzer.comprehensive_analysis(data)
    end_time = time.time()
    
    print(f"Size: {size}, Time: {end_time - start_time:.2f}s")
```

#### 2. Memory Usage Testing
```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Monitor memory during analysis
initial_memory = get_memory_usage()
# ... run analysis ...
final_memory = get_memory_usage()
print(f"Memory used: {final_memory - initial_memory:.2f} MB")
```

## 📊 Test Data

### Sample Files
- `sample_data/sample_sales_data.csv` - 30 rows of sales data
- Generated test data in test files

### Creating Custom Test Data
```python
import pandas as pd
import numpy as np

# Create custom test dataset
def create_test_data(rows=1000):
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=rows, freq='D'),
        'sales': np.random.randint(1000, 5000, rows),
        'profit': np.random.randint(100, 500, rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], rows)
    })

# Save test data
test_data = create_test_data(1000)
test_data.to_csv('my_test_data.csv', index=False)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Make sure you're in the analysis directory
cd analysis

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r requirements.txt
```

#### 2. API Tests Failing
```bash
# Make sure Flask app is running
python app.py

# Check if port 3032 is available
netstat -an | grep 3032

# Test health endpoint manually
curl http://localhost:3032/health
```

#### 3. Performance Issues
```bash
# Check available memory
free -h

# Monitor CPU usage
top -p $(pgrep -f python)

# Reduce test data size
python run_tests.py --performance  # Uses smaller datasets
```

#### 4. Test Data Issues
```bash
# Verify sample data exists
ls -la sample_data/

# Check file permissions
chmod 644 sample_data/sample_sales_data.csv

# Validate CSV format
head -5 sample_data/sample_sales_data.csv
```

### Debug Mode

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Run Tests with Debug Output
```bash
python run_tests.py --verbose
```

#### Check Individual Components
```python
# Test data analyzer directly
from utils.data_analyzer import DataAnalyzer
import pandas as pd

analyzer = DataAnalyzer()
data = pd.read_csv('sample_data/sample_sales_data.csv')
result = analyzer.comprehensive_analysis(data)
print(result)
```

## 📈 Test Coverage

### Generate Coverage Report
```bash
# Install coverage tools
pip install coverage pytest-cov

# Run tests with coverage
coverage run -m unittest discover tests/
coverage report
coverage html  # Creates htmlcov/ directory
```

### View Coverage Report
```bash
# Open coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🎯 Best Practices

### 1. Test Regularly
- Run tests before commits
- Set up automated testing (CI/CD)
- Test with different data types

### 2. Write Good Tests
- Test edge cases
- Use descriptive test names
- Keep tests independent
- Mock external dependencies

### 3. Monitor Performance
- Track performance over time
- Test with realistic data sizes
- Monitor memory usage
- Profile slow operations

### 4. Document Issues
- Record test failures
- Document workarounds
- Update tests for new features
- Keep test data current

## 🚀 Next Steps

1. **Set up Continuous Integration**
   - GitHub Actions
   - Jenkins
   - GitLab CI

2. **Add More Test Types**
   - Security tests
   - Load tests
   - User acceptance tests

3. **Improve Test Coverage**
   - Aim for >90% coverage
   - Test error conditions
   - Test with real data

4. **Automate Testing**
   - Pre-commit hooks
   - Scheduled test runs
   - Performance regression tests

---

## 📞 Need Help?

If you encounter issues:
1. Check the troubleshooting section
2. Review test output carefully
3. Run individual tests to isolate problems
4. Check dependencies and environment setup

Happy testing! 🎉 