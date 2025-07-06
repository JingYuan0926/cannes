import { useState } from 'react';

export default function TestAPI() {
  const [logs, setLogs] = useState(['Ready to test APIs...\n']);

  const log = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = `[${timestamp}] ${message}`;
    setLogs(prev => [...prev, newLog]);
  };

  const clearLog = () => {
    setLogs(['Log cleared...\n']);
  };

  // Test data
  const testData = [
    {"name": "John", "age": 25, "city": "NYC"},
    {"name": "Jane", "age": 30, "city": "LA"}
  ];

  const testETL = async () => {
    log('Testing ETL Service...', 'info');
    try {
      // Create a test CSV file
      const csvContent = 'name,age,city\nJohn,25,NYC\nJane,30,LA';
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const formData = new FormData();
      formData.append('file', blob, 'test.csv');
      formData.append('goal', 'test analysis');

      const response = await fetch('http://localhost:3030/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      log(`ETL Success: Data processed with ${data.processed_data.length} rows`, 'success');
      return data;
    } catch (error) {
      log(`ETL Error: ${error.message}`, 'error');
      throw error;
    }
  };

  const testPreprocessing = async () => {
    log('Testing Preprocessing Service...', 'info');
    try {
      const response = await fetch('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: testData,
          goal: 'test preprocessing'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      log(`Preprocessing Success: ${data.processed_shape[0]} rows processed`, 'success');
      return data;
    } catch (error) {
      log(`Preprocessing Error: ${error.message}`, 'error');
      throw error;
    }
  };

  const testEDA = async () => {
    log('Testing EDA Service...', 'info');
    try {
      const response = await fetch('http://localhost:3035/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: testData,
          prompt: 'test EDA analysis'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      log(`EDA Success: Generated ${data.visualizations ? data.visualizations.length : 0} visualizations`, 'success');
      return data;
    } catch (error) {
      log(`EDA Error: ${error.message}`, 'error');
      throw error;
    }
  };

  const testAnalysis = async () => {
    log('Testing Analysis Service...', 'info');
    try {
      const response = await fetch('http://localhost:3040/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: testData,
          goal: 'test ML analysis'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      log(`Analysis Success: Completed ML analysis`, 'success');
      return data;
    } catch (error) {
      log(`Analysis Error: ${error.message}`, 'error');
      throw error;
    }
  };

  const testFullPipeline = async () => {
    log('=== Testing Full Pipeline ===', 'info');
    try {
      // Step 1: ETL
      log('Step 1: ETL Processing...', 'info');
      const etlData = await testETL();
      
      // Step 2: Preprocessing
      log('Step 2: Preprocessing...', 'info');
      const preprocessData = await fetch('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: etlData.processed_data || etlData.data,
          goal: 'machine learning preparation'
        })
      });

      if (!preprocessData.ok) {
        throw new Error(`Preprocessing failed: ${preprocessData.status}`);
      }

      const preprocessResult = await preprocessData.json();
      log(`Step 2 Success: Preprocessed ${preprocessResult.processed_shape[0]} rows`, 'success');

      // Step 3: EDA
      log('Step 3: EDA Analysis...', 'info');
      const edaData = await fetch('http://localhost:3035/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: preprocessResult.processed_data || preprocessResult.data,
          prompt: 'Comprehensive data analysis'
        })
      });

      if (!edaData.ok) {
        throw new Error(`EDA failed: ${edaData.status}`);
      }

      const edaResult = await edaData.json();
      log(`Step 3 Success: Generated ${edaResult.visualizations ? edaResult.visualizations.length : 0} visualizations`, 'success');

      // Step 4: ML Analysis
      log('Step 4: ML Analysis...', 'info');
      const mlData = await fetch('http://localhost:3040/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          data: preprocessResult.processed_data || preprocessResult.data,
          goal: 'comprehensive analysis'
        })
      });

      if (!mlData.ok) {
        throw new Error(`ML Analysis failed: ${mlData.status}`);
      }

      const mlResult = await mlData.json();
      log(`Step 4 Success: Completed ML analysis`, 'success');

      log('=== Full Pipeline Completed Successfully! ===', 'success');

    } catch (error) {
      log(`Pipeline Error: ${error.message}`, 'error');
    }
  };

  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif',
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      background: '#f5f5f5',
      minHeight: '100vh'
    }}>
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <h1>AI Data Analysis Pipeline - API Test</h1>
        <p>This page tests the API endpoints from the correct origin (localhost:3000).</p>
        
        <div style={{ marginBottom: '20px' }}>
          <button 
            onClick={testETL}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test ETL Service
          </button>
          <button 
            onClick={testPreprocessing}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test Preprocessing Service
          </button>
          <button 
            onClick={testEDA}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test EDA Service
          </button>
          <button 
            onClick={testAnalysis}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test Analysis Service
          </button>
          <button 
            onClick={testFullPipeline}
            style={{
              background: '#28a745',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test Full Pipeline
          </button>
          <button 
            onClick={clearLog}
            style={{
              background: '#6c757d',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Clear Log
          </button>
        </div>
      </div>

      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2>Test Results</h2>
        <div style={{
          background: '#f8f9fa',
          border: '1px solid #dee2e6',
          padding: '15px',
          borderRadius: '4px',
          whiteSpace: 'pre-wrap',
          fontFamily: 'monospace',
          fontSize: '12px',
          maxHeight: '400px',
          overflowY: 'auto'
        }}>
          {logs.map((log, index) => (
            <div key={index} style={{
              color: log.includes('Error') ? '#dc3545' : 
                    log.includes('Success') ? '#28a745' : 
                    log.includes('===') ? '#007bff' : '#000'
            }}>
              {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 