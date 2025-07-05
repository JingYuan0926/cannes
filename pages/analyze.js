import { useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serviceStatus, setServiceStatus] = useState({});
  const [currentStep, setCurrentStep] = useState('');

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
      setCurrentStep('Processing data...');
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
      setCurrentStep('Preprocessing data...');
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
      setCurrentStep('Generating visualizations...');
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
      setCurrentStep('Running machine learning analysis...');
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
      setCurrentStep('Analysis complete!');

    } catch (err) {
      setError(err.message);
      setCurrentStep('');
    } finally {
      setLoading(false);
    }
  };

  // Render simplified insights
  const renderSimpleInsights = (insights) => {
    if (!insights || !insights.ai_insights) return null;

    const allInsights = Object.values(insights.ai_insights);
    const keyFindings = [];
    const recommendations = [];
    const actualInsights = [];

    allInsights.forEach((category, index) => {
      // Handle the nested structure: category.insights contains JSON strings
      if (category && category.insights && Array.isArray(category.insights)) {
        category.insights.forEach(insightString => {
          if (typeof insightString === 'string') {
            try {
              // Remove markdown code block formatting if present
              let cleanJson = insightString;
              if (cleanJson.includes('```json')) {
                cleanJson = cleanJson.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim();
              }
              
              // Parse the JSON string
              const parsedInsight = JSON.parse(cleanJson);
              
              // Extract data from parsed object
              if (parsedInsight.insights && Array.isArray(parsedInsight.insights)) {
                actualInsights.push(...parsedInsight.insights);
              }
              
              if (parsedInsight.key_findings && Array.isArray(parsedInsight.key_findings)) {
                keyFindings.push(...parsedInsight.key_findings);
              }
              
              if (parsedInsight.recommendations && Array.isArray(parsedInsight.recommendations)) {
                recommendations.push(...parsedInsight.recommendations);
              }
            } catch (e) {
              console.warn('Could not parse insight JSON:', insightString);
            }
          }
        });
      }
    });

    // Filter and deduplicate meaningful content
    const getUniqueFiltered = (arr) => {
      const filtered = arr.filter(item => 
        item && 
        typeof item === 'string' &&
        item !== "AI analysis completed" && 
        !item.includes("Review the insights provided") &&
        item.length > 30
      );
      // Remove duplicates
      return [...new Set(filtered)];
    };

    const filteredInsights = getUniqueFiltered(actualInsights);
    const filteredFindings = getUniqueFiltered(keyFindings);
    const filteredRecommendations = getUniqueFiltered(recommendations);

    return (
      <div className="simple-insights">
        {filteredInsights.length > 0 && (
          <div className="insight-section">
            <h3>🧠 AI Insights</h3>
            <ul>
              {filteredInsights.slice(0, 8).map((insight, idx) => (
                <li key={idx}>{insight}</li>
              ))}
            </ul>
          </div>
        )}

        {filteredFindings.length > 0 && (
          <div className="insight-section">
            <h3>🔍 Key Findings</h3>
            <ul>
              {filteredFindings.slice(0, 6).map((finding, idx) => (
                <li key={idx}>{finding}</li>
              ))}
            </ul>
          </div>
        )}
        
        {filteredRecommendations.length > 0 && (
          <div className="insight-section">
            <h3>💡 Recommendations</h3>
            <ul>
              {filteredRecommendations.slice(0, 6).map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {filteredInsights.length === 0 && filteredFindings.length === 0 && filteredRecommendations.length === 0 && (
          <div className="no-insights">
            <p>🤖 Analysis in progress - detailed insights will appear here once processing is complete.</p>
          </div>
        )}
      </div>
    );
  };

  // Render ML analysis results (simplified)
  const renderMLResults = (mlData) => {
    if (!mlData) return null;

    const results = mlData.results || mlData;
    const analyses = results.analyses || [];

    return (
      <div className="ml-results">
        <h2>🤖 Machine Learning Analysis</h2>
        
        {analyses && analyses.length > 0 ? (
          <div className="analyses">
            {analyses.map((analysisItem, idx) => (
              <div key={idx} className="analysis-item">
                <h3>{analysisItem.algorithm} - {analysisItem.analysis_type}</h3>
                
                {/* Analysis Insights */}
                {analysisItem.insights && analysisItem.insights.length > 0 && (
                  <div className="ml-insights">
                    <h4>💡 Insights:</h4>
                    <ul>
                      {analysisItem.insights.map((insight, insightIdx) => (
                        <li key={insightIdx}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Visualizations/Graphs */}
                {analysisItem.graphs && analysisItem.graphs.length > 0 && (
                  <div className="ml-visualizations">
                    <div className="charts-grid">
                      {analysisItem.graphs
                        .filter(graph => !graph.title || !graph.title.toLowerCase().includes('solution distribution'))
                        .map((graph, graphIdx) => (
                        <div key={graphIdx} className="chart-container">
                          <h4>{graph.title || `Chart ${graphIdx + 1}`}</h4>
                          
                          {/* Render Plotly Chart */}
                          {graph.data && (
                            <div className="chart-wrapper">
                              {Plot && (() => {
                                try {
                                  const chartData = JSON.parse(graph.data);
                                  return (
                                    <Plot
                                      data={chartData.data}
                                      layout={{
                                        ...chartData.layout,
                                        width: undefined,
                                        height: 400,
                                        margin: { t: 40, r: 20, b: 50, l: 60 }
                                      }}
                                      config={{ 
                                        displayModeBar: true,
                                        displaylogo: false,
                                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                      }}
                                      style={{ width: '100%', height: '100%' }}
                                      onError={(err) => console.error('ML Chart error:', err)}
                                    />
                                  );
                                } catch (err) {
                                  console.error('Error parsing ML chart data:', err);
                                  return (
                                    <div className="chart-error">
                                      ⚠️ Chart could not be displayed
                                    </div>
                                  );
                                }
                              })()}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="no-ml-results">
            <h4>⚠️ No ML Analyses Available</h4>
            <p>The dataset may not have suitable features for machine learning analysis.</p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="analyze-page">
      <h1>🚀 AI-Powered Data Analysis Pipeline</h1>
      
      <div className="service-status-section">
        <h2>Service Status</h2>
        <button onClick={checkServices} className="check-services-btn">
          Check Services
        </button>
        <div className="services-grid">
          {Object.entries(serviceStatus).map(([name, status]) => (
            <div key={name} className={`service-card ${status.healthy ? 'healthy' : 'down'}`}>
              <strong>{name.toUpperCase()}</strong><br />
              {status.healthy ? '✅ Healthy' : '❌ Down'}<br />
              {status.status && <small>{status.status}</small>}
              {status.error && <small className="error-text">{status.error}</small>}
            </div>
          ))}
        </div>
      </div>

      <div className="upload-section">
        <h2>Upload & Analyze Data</h2>
        <div className="file-input-container">
          <input 
            type="file" 
            accept=".csv,.xlsx,.json"
            onChange={(e) => setFile(e.target.files[0])}
            className="file-input"
          />
          {file && <p className="file-selected">Selected: {file.name}</p>}
        </div>
        
        <div className="prompt-container">
          <textarea 
            placeholder="Enter your analysis goals (e.g., 'Find transaction patterns', 'Predict customer behavior', etc.)"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="prompt-textarea"
          />
        </div>
        
        <button 
          onClick={handleAnalyze} 
          disabled={loading || !file}
          className="analyze-btn"
        >
          {loading ? `${currentStep || 'Analyzing...'}` : 'Analyze Data'}
        </button>
      </div>

      {error && (
        <div className="error-section">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          {/* EDA Results */}
          {results.eda?.analysis && (
            <div className="eda-section">
              <h2>📈 Exploratory Data Analysis</h2>
              
              {/* Visualizations Only */}
              {results.eda.analysis.visualizations && results.eda.analysis.visualizations.length > 0 && (
                <div className="visualizations-section">
                  <div className="charts-grid">
                    {results.eda.analysis.visualizations.map((chart, index) => (
                      <div key={index} className="chart-container">
                        <h3>{chart.title}</h3>
                        <p className="chart-description">{chart.description}</p>
                        <div className="chart-badge">
                          {chart.category} • {chart.chart_type}
                        </div>
                        
                        {chart.chart_json && (
                          <div className="chart-wrapper">
                            {Plot ? (
                              <Plot
                                data={JSON.parse(chart.chart_json).data}
                                layout={{
                                  ...JSON.parse(chart.chart_json).layout,
                                  autosize: true,
                                  height: 400,
                                  margin: { l: 50, r: 50, t: 50, b: 50 }
                                }}
                                config={{ displayModeBar: true, responsive: true }}
                                style={{ width: '100%', height: '400px' }}
                                onError={(err) => console.error('Plotly error:', err)}
                              />
                            ) : (
                              <div className="chart-loading">
                                📊 Loading chart...
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Simplified EDA Insights */}
              {results.eda.analysis.insights && (
                <div className="insights-section">
                  {renderSimpleInsights(results.eda.analysis.insights)}
                </div>
              )}
            </div>
          )}

          {/* ML Analysis Results */}
          {results.ml && (
            <div className="ml-section">
              {renderMLResults(results.ml)}
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .analyze-page {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .service-status-section {
          margin-bottom: 30px;
          padding: 20px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #f8f9fa;
        }

        .check-services-btn {
          margin-bottom: 15px;
          padding: 8px 16px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .services-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
        }

        .service-card {
          padding: 15px;
          border-radius: 8px;
          text-align: center;
        }

        .service-card.healthy {
          border: 2px solid #4CAF50;
          background: #e8f5e8;
        }

        .service-card.down {
          border: 2px solid #f44336;
          background: #ffeaea;
        }

        .error-text {
          color: #f44336;
        }

        .upload-section {
          margin-bottom: 30px;
          padding: 20px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: white;
        }

        .file-input {
          margin-bottom: 10px;
        }

        .file-selected {
          color: #28a745;
          font-weight: 500;
        }

        .prompt-textarea {
          width: 100%;
          padding: 10px;
          border-radius: 4px;
          border: 1px solid #ccc;
          margin-bottom: 15px;
          font-family: inherit;
        }

        .analyze-btn {
          padding: 12px 24px;
          font-size: 16px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .analyze-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .error-section {
          padding: 20px;
          background: #ffeaea;
          border: 1px solid #f44336;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .error-section h3 {
          color: #f44336;
          margin-top: 0;
        }

        .eda-section {
          margin-bottom: 40px;
        }

        .charts-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
          gap: 30px;
          margin: 30px 0;
        }

        .chart-container {
          border: 1px solid #ddd;
          border-radius: 12px;
          padding: 20px;
          background: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .chart-container h3, .chart-container h4 {
          margin-top: 0;
          color: #333;
        }

        .chart-description {
          color: #666;
          font-size: 14px;
          margin-bottom: 10px;
        }

        .chart-badge {
          display: inline-block;
          padding: 4px 12px;
          background: #e3f2fd;
          border-radius: 16px;
          font-size: 12px;
          color: #1976d2;
          margin-bottom: 15px;
        }

        .chart-wrapper {
          min-height: 400px;
          width: 100%;
        }

        .chart-loading {
          padding: 20px;
          text-align: center;
          background: #f8f9fa;
          border: 1px dashed #ddd;
          border-radius: 4px;
          color: #666;
        }

        .chart-error {
          padding: 20px;
          text-align: center;
          background: #fff3cd;
          border: 1px dashed #ffeeba;
          border-radius: 4px;
          color: #856404;
        }

        .insights-section {
          margin-top: 30px;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .simple-insights .insight-section {
          margin-bottom: 25px;
        }

        .simple-insights h3 {
          color: #007bff;
          margin-bottom: 15px;
        }

        .simple-insights ul {
          list-style: none;
          padding: 0;
        }

        .simple-insights li {
          padding: 8px 0;
          border-bottom: 1px solid #e9ecef;
          line-height: 1.5;
        }

        .simple-insights li:last-child {
          border-bottom: none;
        }

        .ml-section {
          margin-top: 40px;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .analysis-item {
          margin-bottom: 30px;
          padding: 20px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: white;
        }

        .ml-insights {
          margin-bottom: 20px;
        }

        .ml-insights h4 {
          color: #28a745;
          margin-bottom: 10px;
        }

        .ml-insights ul {
          list-style: none;
          padding: 0;
        }

        .ml-insights li {
          padding: 5px 0;
          line-height: 1.4;
        }

        .no-ml-results {
          padding: 20px;
          text-align: center;
          background: #fff3cd;
          border: 1px solid #ffeeba;
          border-radius: 8px;
          color: #856404;
        }

        .no-insights {
          padding: 15px;
          text-align: center;
          background: #e3f2fd;
          border: 1px solid #bbdefb;
          border-radius: 8px;
          color: #1565c0;
          font-style: italic;
        }
      `}</style>
    </div>
  );
}
