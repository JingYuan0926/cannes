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

  // Render insights in a formatted way
  const renderInsights = (insights) => {
    if (!insights) return null;

    return (
      <div className="insights-container">
        {insights.ai_insights && (
          <div className="ai-insights">
            <h3>🤖 AI Insights</h3>
            {Object.entries(insights.ai_insights).map(([category, data]) => (
              <div key={category} className="insight-category">
                <h4>{category.replace('_', ' ').toUpperCase()}</h4>
                {data.insights && (
                  <ul>
                    {data.insights.map((insight, idx) => (
                      <li key={idx}>{insight}</li>
                    ))}
                  </ul>
                )}
                {data.key_findings && data.key_findings.length > 0 && (
                  <div>
                    <strong>Key Findings:</strong>
                    <ul>
                      {data.key_findings.map((finding, idx) => (
                        <li key={idx}>{finding}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {data.recommendations && data.recommendations.length > 0 && (
                  <div>
                    <strong>Recommendations:</strong>
                    <ul>
                      {data.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {insights.dataset_summary && (
          <div className="dataset-summary">
            <h3>📊 Dataset Summary</h3>
            <p><strong>Shape:</strong> {insights.dataset_summary.shape?.[0]} rows × {insights.dataset_summary.shape?.[1]} columns</p>
            <p><strong>Columns:</strong> {insights.dataset_summary.columns?.join(', ')}</p>
          </div>
        )}

        {insights.key_findings && insights.key_findings.length > 0 && (
          <div className="key-findings">
            <h3>🔍 Key Findings</h3>
            <ul>
              {insights.key_findings.map((finding, idx) => (
                <li key={idx}>{finding}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  // Render ML analysis results
  const renderMLResults = (mlData) => {
    if (!mlData) return null;

    console.log('ML Data Structure:', mlData); // Debug log

    // Handle the actual data structure from analysis service
    const results = mlData.results || mlData;
    const analyses = results.analyses || [];
    const analyticsummary = results.analytics_summary || {};
    const overallInsights = results.overall_insights || {};

    return (
      <div className="ml-results">
        <h2>🤖 Machine Learning Analysis</h2>
        
        {/* Analytics Summary */}
        <div className="analytics-summary">
          <h3>📊 Analysis Summary</h3>
          <p><strong>Analysis ID:</strong> {mlData.analysis_id || 'N/A'}</p>
          <p><strong>Status:</strong> {mlData.message || 'Unknown'}</p>
          <p><strong>Total Analyses Performed:</strong> {analyses.length || 0}</p>
          <p><strong>Completed At:</strong> {results.completed_at || 'N/A'}</p>
        </div>

        {/* Analysis Results */}
        {analyses && analyses.length > 0 ? (
          <div className="analyses">
            <h3>🔬 Analysis Results</h3>
            {analyses.map((analysisItem, idx) => (
              <div key={idx} className="analysis-item">
                <h4>
                  {analysisItem.algorithm} ({analysisItem.analysis_type})
                  <span style={{ 
                    fontSize: '12px', 
                    backgroundColor: '#e3f2fd', 
                    padding: '2px 8px', 
                    borderRadius: '12px',
                    marginLeft: '10px'
                  }}>
                    Priority: {analysisItem.config?.priority || 'N/A'}
                  </span>
                </h4>
                
                {/* Analysis Config */}
                {analysisItem.config && (
                  <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
                    <strong>Analysis Purpose:</strong> {analysisItem.config.expected_insights || 'N/A'}<br/>
                    <strong>Justification:</strong> {analysisItem.config.justification || 'N/A'}
                  </div>
                )}

                {/* Analysis Insights */}
                {analysisItem.insights && analysisItem.insights.length > 0 && (
                  <div style={{ marginBottom: '15px' }}>
                    <h5>💡 Key Insights:</h5>
                    <ul>
                      {analysisItem.insights.map((insight, insightIdx) => (
                        <li key={insightIdx}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Analysis Results */}
                {analysisItem.results && (
                  <div style={{ marginBottom: '15px' }}>
                    <h5>📈 Results:</h5>
                    <div className="analysis-results">
                      {typeof analysisItem.results === 'object' ? (
                        <pre>{JSON.stringify(analysisItem.results, null, 2)}</pre>
                      ) : (
                        <p>{analysisItem.results}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Visualizations/Graphs */}
                {analysisItem.graphs && analysisItem.graphs.length > 0 && (
                  <div className="ml-visualizations">
                    <h5>📊 Visualizations:</h5>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>
                      {analysisItem.graphs.map((graph, graphIdx) => (
                        <div key={graphIdx} style={{ 
                          border: '1px solid #ddd', 
                          borderRadius: '8px', 
                          padding: '15px',
                          backgroundColor: 'white'
                        }}>
                          <h6>{graph.title || `Chart ${graphIdx + 1}`}</h6>
                          <p style={{ color: '#666', fontSize: '12px' }}>Type: {graph.type}</p>
                          
                          {/* Render Plotly Chart */}
                          {graph.data && (
                            <div style={{ width: '100%', height: '400px' }}>
                              {Plot && (() => {
                                try {
                                  const chartData = JSON.parse(graph.data);
                                  return (
                                    <Plot
                                      data={chartData.data}
                                      layout={{
                                        ...chartData.layout,
                                        width: undefined,
                                        height: 350,
                                        margin: { t: 30, r: 20, b: 40, l: 60 }
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
                                    <div style={{ 
                                      padding: '20px', 
                                      textAlign: 'center', 
                                      backgroundColor: '#fff3cd',
                                      border: '1px dashed #ffeeba',
                                      borderRadius: '4px',
                                      color: '#856404'
                                    }}>
                                      ⚠️ Chart data could not be displayed
                                      <br />
                                      <small>Check console for details</small>
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
          <div style={{ 
            padding: '20px', 
            textAlign: 'center', 
            backgroundColor: '#fff3cd', 
            border: '1px solid #ffeeba',
            borderRadius: '8px',
            color: '#856404'
          }}>
            <h4>⚠️ No ML Analyses Performed</h4>
            <p>The dataset may not have enough data or suitable features for machine learning analysis.</p>
            <p>Try uploading a larger dataset with more numerical features.</p>
          </div>
        )}

        {/* Overall Insights */}
        {overallInsights && Object.keys(overallInsights).length > 0 && (
          <div className="overall-insights">
            <h3>🎯 Overall Insights</h3>
            
            {overallInsights.conclusions && overallInsights.conclusions.length > 0 && (
              <div style={{ marginBottom: '15px' }}>
                <h4>📋 Conclusions:</h4>
                <ul>
                  {overallInsights.conclusions.map((conclusion, idx) => (
                    <li key={idx}>{conclusion}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {overallInsights.recommendations && overallInsights.recommendations.length > 0 && (
              <div style={{ marginBottom: '15px' }}>
                <h4>💡 Recommendations:</h4>
                <ul>
                  {overallInsights.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {overallInsights.strategy && (
              <div style={{ marginBottom: '15px' }}>
                <h4>🎯 Strategy:</h4>
                <p>{overallInsights.strategy}</p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>🚀 AI-Powered Data Analysis Pipeline</h1>
      
      <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h2>Service Status</h2>
        <button onClick={checkServices} style={{ marginBottom: '10px', padding: '8px 16px' }}>
          Check Services
        </button>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
          {Object.entries(serviceStatus).map(([name, status]) => (
            <div key={name} style={{ 
              padding: '10px', 
              border: `2px solid ${status.healthy ? '#4CAF50' : '#f44336'}`,
              borderRadius: '4px',
              backgroundColor: status.healthy ? '#e8f5e8' : '#ffeaea'
            }}>
              <strong>{name.toUpperCase()}</strong><br />
              {status.healthy ? '✅ Healthy' : '❌ Down'}<br />
              {status.status && <small>{status.status}</small>}
              {status.error && <small style={{ color: 'red' }}>{status.error}</small>}
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h2>Upload & Analyze Data</h2>
        <div style={{ marginBottom: '15px' }}>
          <input 
            type="file" 
            accept=".csv,.xlsx,.json"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ marginBottom: '10px' }}
          />
          {file && <p>Selected: {file.name}</p>}
        </div>
        
        <div style={{ marginBottom: '15px' }}>
          <textarea 
            placeholder="Enter your analysis goals (e.g., 'Find transaction patterns', 'Predict customer behavior', etc.)"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
        </div>
        
        <button 
          onClick={handleAnalyze} 
          disabled={loading || !file}
          style={{ 
            padding: '12px 24px', 
            fontSize: '16px', 
            backgroundColor: loading ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? `${currentStep || 'Analyzing...'}` : 'Analyze Data'}
        </button>
      </div>

      {error && (
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#ffeaea', 
          border: '1px solid #f44336', 
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#f44336' }}>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div>
          {/* EDA Results */}
          {results.eda?.analysis && (
            <div style={{ marginBottom: '40px' }}>
              <h2>📈 Exploratory Data Analysis (EDA)</h2>
              
              {/* Dataset Info */}
              {results.eda.analysis.dataset_info && (
                <div style={{ 
                  padding: '15px', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '8px',
                  marginBottom: '20px'
                }}>
                  <h3>Dataset Information</h3>
                  <p><strong>Shape:</strong> {results.eda.analysis.dataset_info.shape?.[0]} rows × {results.eda.analysis.dataset_info.shape?.[1]} columns</p>
                  <p><strong>Columns:</strong> {results.eda.analysis.dataset_info.columns?.join(', ')}</p>
                </div>
              )}

              {/* Visualizations */}
              {results.eda.analysis.visualizations && results.eda.analysis.visualizations.length > 0 && (
                <div>
                  <h3>📊 Generated Visualizations</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(600px, 1fr))', gap: '20px' }}>
                    {results.eda.analysis.visualizations.map((chart, index) => (
                      <div key={index} style={{ 
                        border: '1px solid #ddd', 
                        borderRadius: '8px', 
                        padding: '20px',
                        backgroundColor: 'white'
                      }}>
                        <h4>{chart.title}</h4>
                        <p style={{ color: '#666', fontSize: '14px' }}>{chart.description}</p>
                        <div style={{ 
                          display: 'inline-block', 
                          padding: '4px 8px', 
                          backgroundColor: '#e3f2fd', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginBottom: '15px'
                        }}>
                          {chart.category} • {chart.chart_type}
                        </div>
                        
                        {chart.chart_json && (
                          <div style={{ minHeight: '400px' }}>
                            {Plot ? (
                              <Plot
                                data={JSON.parse(chart.chart_json).data}
                                layout={{
                                  ...JSON.parse(chart.chart_json).layout,
                                  autosize: true,
                                  margin: { l: 50, r: 50, t: 50, b: 50 }
                                }}
                                config={{ displayModeBar: true, responsive: true }}
                                style={{ width: '100%', height: '400px' }}
                                onError={(err) => console.error('Plotly error:', err)}
                              />
                            ) : (
                              <div style={{ 
                                padding: '20px', 
                                textAlign: 'center', 
                                backgroundColor: '#f8f9fa',
                                border: '1px dashed #ddd',
                                borderRadius: '4px'
                              }}>
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

              {/* EDA Insights */}
              {results.eda.analysis.insights && (
                <div style={{ marginTop: '30px' }}>
                  <h3>💡 EDA Insights</h3>
                  <div style={{ 
                    padding: '20px', 
                    backgroundColor: '#f8f9fa', 
                    borderRadius: '8px' 
                  }}>
                    {renderInsights(results.eda.analysis.insights)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ML Analysis Results */}
          {results.ml && (
            <div style={{ 
              marginTop: '40px',
              padding: '20px', 
              backgroundColor: '#f8f9fa', 
              borderRadius: '8px' 
            }}>
              {renderMLResults(results.ml)}
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .insights-container {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .insight-category {
          margin-bottom: 20px;
          padding: 15px;
          border-left: 4px solid #007bff;
          background: white;
          border-radius: 0 8px 8px 0;
        }
        .insight-category h4 {
          margin-top: 0;
          color: #007bff;
          font-size: 16px;
        }
        .insight-category ul {
          margin: 10px 0;
          padding-left: 20px;
        }
        .insight-category li {
          margin-bottom: 8px;
          line-height: 1.5;
        }
        .analysis-item {
          margin-bottom: 20px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: white;
        }
        .analysis-results {
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          font-size: 12px;
          overflow-x: auto;
        }
        .ml-visualization {
          margin-top: 10px;
          padding: 10px;
          background: #e8f5e8;
          border-radius: 4px;
        }
        .dataset-summary, .key-findings, .analytics-summary, .overall-insights {
          margin-bottom: 20px;
        }
        .dataset-summary h3, .key-findings h3, .analytics-summary h3, .overall-insights h3 {
          margin-bottom: 15px;
          color: #333;
        }
      `}</style>
    </div>
  );
}
