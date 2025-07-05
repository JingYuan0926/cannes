import { useState } from 'react';

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serviceStatus, setServiceStatus] = useState({});

  // Check service health
  const checkServices = async () => {
    const services = {
      etl: 'http://localhost:3030',
      preprocessing: 'http://localhost:3031', 
      eda: 'http://localhost:3035',
      analysis: 'http://localhost:3040'
    };

    const status = {};
    for (const [name, url] of Object.entries(services)) {
      try {
        const response = await fetch(url);
        const data = await response.json();
        status[name] = { healthy: true, status: data.status };
      } catch (err) {
        status[name] = { healthy: false, error: err.message };
      }
    }
    setServiceStatus(status);
  };

  // Handle file upload and analysis
  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);
      formData.append('goal', prompt || 'data analysis');

      // Step 1: ETL Service (File Upload)
      console.log('Starting ETL...');
      
      const etlResponse = await fetch('http://localhost:3030/analyze', {
        method: 'POST',
        body: formData
      });
      
      if (!etlResponse.ok) {
        throw new Error('ETL processing failed');
      }
      
      const etlData = await etlResponse.json();
      console.log('ETL Result:', etlData);

      // Step 2: Preprocessing Service
      console.log('Starting Preprocessing...');
      const preprocessResponse = await fetch('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: etlData.processed_data || etlData.data,
          goal: prompt || 'machine learning preparation'
        })
      });

      if (!preprocessResponse.ok) {
        throw new Error('Preprocessing failed');
      }

      const preprocessData = await preprocessResponse.json();
      console.log('Preprocessing Result:', preprocessData);

      // Step 3: EDA Service
      console.log('Starting EDA...');
      const edaResponse = await fetch('http://localhost:3035/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          prompt: prompt || 'Comprehensive data analysis'
        })
      });

      if (!edaResponse.ok) {
        throw new Error('EDA analysis failed');
      }

      const edaData = await edaResponse.json();
      console.log('EDA Result:', edaData);

      // Step 4: ML Analysis Service
      console.log('Starting ML Analysis...');
      const mlResponse = await fetch('http://localhost:3040/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          goal: prompt || 'comprehensive analysis'
        })
      });

      if (!mlResponse.ok) {
        throw new Error('ML analysis failed');
      }

      const mlData = await mlResponse.json();
      console.log('ML Result:', mlData);

      // Combine all results
      setResults({
        etl: etlData,
        preprocessing: preprocessData,
        eda: edaData,
        ml: mlData
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Data Analysis Pipeline Test</h1>
      
      <div>
        <h2>Service Status</h2>
        <button onClick={checkServices}>Check Services</button>
        {Object.entries(serviceStatus).map(([name, status]) => (
          <div key={name}>
            {name}: {status.healthy ? '✅ Healthy' : '❌ Down'} 
            {status.status && ` - ${status.status}`}
            {status.error && ` - Error: ${status.error}`}
          </div>
        ))}
      </div>

      <div>
        <h2>Upload File</h2>
        <input 
          type="file" 
          accept=".csv,.xlsx,.json"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <br />
        <textarea 
          placeholder="Analysis prompt (optional)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          cols={50}
        />
        <br />
        <button onClick={handleAnalyze} disabled={loading || !file}>
          {loading ? 'Analyzing...' : 'Analyze Data'}
        </button>
      </div>

      {error && (
        <div>
          <h3>Error:</h3>
          <pre>{error}</pre>
        </div>
      )}

      {results && (
        <div>
          <h2>Results</h2>
          
          <h3>ETL Results</h3>
          <pre>{JSON.stringify(results.etl, null, 2)}</pre>
          
          <h3>Preprocessing Results</h3>
          <pre>{JSON.stringify(results.preprocessing, null, 2)}</pre>
          
          <h3>EDA Results</h3>
          <pre>{JSON.stringify(results.eda, null, 2)}</pre>
          
          <h3>ML Results</h3>
          <pre>{JSON.stringify(results.ml, null, 2)}</pre>

          {results.eda?.analysis?.visualizations && (
            <div>
              <h3>Chart Data (JSON Format)</h3>
              {results.eda.analysis.visualizations.map((chart, index) => (
                <div key={index}>
                  <h4>{chart.title}</h4>
                  <p>{chart.description}</p>
                  <p>Type: {chart.chart_type}</p>
                  <p>Category: {chart.category}</p>
                  <details>
                    <summary>Chart JSON Data</summary>
                    <pre>{JSON.stringify(chart.chart_json, null, 2)}</pre>
                  </details>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
