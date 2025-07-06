import { useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { motion } from 'framer-motion';

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
  const [logs, setLogs] = useState([]);

  // Add logging function
  const log = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = `[${timestamp}] ${message}`;
    setLogs(prev => [...prev, newLog]);
    console.log(newLog);
  };

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
        const response = await fetch(url, {
          method: 'GET',
          timeout: 5000
        });
        const data = await response.json();
        status[name] = { healthy: true, status: data.status };
      } catch (err) {
        status[name] = { healthy: false, error: err.message };
      }
    }
    setServiceStatus(status);
  };

  // Improved fetch with timeout and better error handling
  const fetchWithTimeout = async (url, options = {}, timeout = 30000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }
      throw error;
    }
  };

  // Handle file upload and analysis
  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResults(null);
    setLogs([]);

    try {
      log('=== Starting AI Data Analysis Pipeline ===', 'info');
      
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);
      formData.append('goal', prompt || 'data analysis');

      // Step 1: ETL Service (File Upload)
      setCurrentStep('Processing data...');
      log('Step 1: ETL Processing...', 'info');
      
      const etlResponse = await fetchWithTimeout('http://localhost:3030/analyze', {
        method: 'POST',
        body: formData
      }, 30000);
      
      const etlData = await etlResponse.json();
      log(`ETL Success: Data processed with ${etlData.processed_data ? etlData.processed_data.length : 'unknown'} rows`, 'success');

      // Step 2: Preprocessing Service
      setCurrentStep('Preprocessing data...');
      log('Step 2: Preprocessing...', 'info');
      
      const preprocessResponse = await fetchWithTimeout('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: etlData.processed_data || etlData.data,
          goal: prompt || 'machine learning preparation'
        })
      }, 30000);

      const preprocessData = await preprocessResponse.json();
      log(`Step 2 Success: Preprocessed ${preprocessData.processed_shape ? preprocessData.processed_shape[0] : 'unknown'} rows`, 'success');

      // Step 3: EDA Service
      setCurrentStep('Generating visualizations...');
      log('Step 3: EDA Analysis...', 'info');
      
      const edaResponse = await fetchWithTimeout('http://localhost:3035/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          prompt: prompt || 'Comprehensive data analysis'
        })
      }, 45000); // Longer timeout for EDA

      const edaData = await edaResponse.json();
      log(`Step 3 Success: Generated ${edaData.analysis && edaData.analysis.visualizations ? edaData.analysis.visualizations.length : 0} visualizations`, 'success');

      // Step 4: ML Analysis Service
      setCurrentStep('Running machine learning analysis...');
      log('Step 4: ML Analysis...', 'info');
      
      const mlResponse = await fetchWithTimeout('http://localhost:3040/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          goal: prompt || 'comprehensive analysis'
        })
      }, 60000); // Longer timeout for ML

      const mlData = await mlResponse.json();
      log('Step 4 Success: Completed ML analysis', 'success');

      // Combine all results
      setResults({
        etl: etlData,
        preprocessing: preprocessData,
        eda: edaData,
        ml: mlData
      });
      
      setCurrentStep('Analysis complete!');
      log('=== Full Pipeline Completed Successfully! ===', 'success');

    } catch (err) {
      const errorMsg = err.message || 'Unknown error occurred';
      setError(errorMsg);
      log(`Pipeline Error: ${errorMsg}`, 'error');
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
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="insights-container"
      >
        {filteredInsights.length > 0 && (
          <div className="insight-card">
            <h3>🧠 AI Insights</h3>
            <ul>
              {filteredInsights.slice(0, 8).map((insight, idx) => (
                <li key={idx}>{insight}</li>
              ))}
            </ul>
          </div>
        )}

        {filteredFindings.length > 0 && (
          <div className="insight-card">
            <h3>🔍 Key Findings</h3>
            <ul>
              {filteredFindings.slice(0, 6).map((finding, idx) => (
                <li key={idx}>{finding}</li>
              ))}
            </ul>
          </div>
        )}
        
        {filteredRecommendations.length > 0 && (
          <div className="insight-card">
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
      </motion.div>
    );
  };

  // Render TEE attestation information
  const renderTEEAttestation = (mlData) => {
    if (!mlData || !mlData.results || !mlData.results.tee_attestation) return null;

    const attestation = mlData.results.tee_attestation;
    
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="tee-attestation-card"
      >
        <h3>🔐 TEE Attestation</h3>
        <div className="attestation-content">
          <div className="attestation-status">
            {attestation.tee_attested ? (
              <div className="status-success">
                <span className="status-icon">✅</span>
                <div>
                  <strong>TEE Verified</strong>
                  <p>This analysis was executed and signed in a Trusted Execution Environment</p>
                </div>
              </div>
            ) : (
              <div className="status-failed">
                <span className="status-icon">⚠️</span>
                <div>
                  <strong>TEE Unavailable</strong>
                  <p>Analysis completed but TEE attestation failed: {attestation.error}</p>
                </div>
              </div>
            )}
          </div>
          
          {attestation.tee_attested && (
            <div className="attestation-metadata">
              <div className="metadata-grid">
                <div className="metadata-item">
                  <label>ROFL App ID:</label>
                  <code>{attestation.rofl_app_id}</code>
                </div>
                <div className="metadata-item">
                  <label>Results Hash:</label>
                  <code>{attestation.results_hash.substring(0, 16)}...</code>
                </div>
                <div className="metadata-item">
                  <label>Algorithm:</label>
                  <span>{attestation.signature_algorithm}</span>
                </div>
                <div className="metadata-item">
                  <label>Timestamp:</label>
                  <span>{new Date(attestation.timestamp).toLocaleString()}</span>
                </div>
              </div>
              
              <div className="verification-info">
                <h4>🔍 Verification Details</h4>
                <p>
                  <strong>Integrity:</strong> Results hash ensures data hasn't been tampered with<br/>
                  <strong>Authenticity:</strong> Signing key proves this came from the TEE<br/>
                  <strong>Non-repudiation:</strong> ROFL app ID provides cryptographic proof of origin
                </p>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  // Render ML analysis results
  const renderMLResults = (mlData) => {
    if (!mlData) return null;

    const results = mlData.results || mlData;
    const analyses = results.analyses || [];

    return (
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="ml-results-section"
      >
        <h2>🤖 Machine Learning Analysis</h2>
        
        {/* TEE Attestation Section */}
        {renderTEEAttestation(mlData)}
        
        {analyses && analyses.length > 0 ? (
          <div className="analyses-container">
            {analyses.map((analysisItem, idx) => (
              <motion.div 
                key={idx} 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: idx * 0.1 }}
                className="analysis-card"
              >
                <h3>{analysisItem.algorithm} - {analysisItem.analysis_type}</h3>
                
                {/* Analysis Insights */}
                {analysisItem.insights && analysisItem.insights.length > 0 && (
                  <div className="analysis-insights">
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
                  <div className="analysis-charts">
                    {analysisItem.graphs
                      .filter(graph => !graph.title || !graph.title.toLowerCase().includes('solution distribution'))
                      .map((graph, graphIdx) => (
                      <div key={graphIdx} className="chart-card">
                        <h4>{graph.title || `Chart ${graphIdx + 1}`}</h4>
                        
                        {/* Render Plotly Chart */}
                        {graph.data && (
                          <div className="chart-container">
                            {Plot && (() => {
                              try {
                                const chartData = JSON.parse(graph.data);
                                return (
                                  <Plot
                                    data={chartData.data}
                                    layout={{
                                      ...chartData.layout,
                                      width: 580,
                                      height: 350,
                                      margin: { t: 30, r: 20, b: 40, l: 50 },
                                      font: { size: 11 }
                                    }}
                                    config={{ 
                                      displayModeBar: false,
                                      responsive: true
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
                )}
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="no-ml-results">
            <h4>⚠️ No ML Analyses Available</h4>
            <p>The dataset may not have suitable features for machine learning analysis.</p>
          </div>
        )}
      </motion.div>
    );
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.4,
        ease: "easeOut",
      },
    },
  };

  return (
    <div className="analyze-page">
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="nav-bar"
      >
        <div className="nav-container">
          <Link href="/analyze">
            <div className="nav-item nav-active">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="nav-item">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="nav-item">
              View
            </div>
          </Link>
        </div>
      </motion.nav>

      {/* Page Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
        className="page-header"
      >
        <h1>🚀 AI-Powered Data Analysis Pipeline</h1>
        <p>Upload your data and let our AI analyze it with advanced machine learning algorithms</p>
      </motion.div>

      {/* Main Content */}
      <motion.main 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="main-content"
      >
        <div className="content-container">
          
          {/* Service Status Section */}
          <motion.div variants={itemVariants} className="service-status-card">
            <h2>Service Status</h2>
            <button onClick={checkServices} className="check-services-btn">
              Check Services
            </button>
            <div className="services-grid">
              {Object.entries(serviceStatus).map(([name, status]) => (
                <div key={name} className={`service-item ${status.healthy ? 'healthy' : 'down'}`}>
                  <strong>{name.toUpperCase()}</strong>
                  <span className="status-indicator">
                    {status.healthy ? '✅ Healthy' : '❌ Down'}
                  </span>
                  {status.status && <small>{status.status}</small>}
                  {status.error && <small className="error-text">{status.error}</small>}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Upload Section */}
          <motion.div variants={itemVariants} className="upload-card">
            <h2>Upload & Analyze Data</h2>
            
            <div className="file-upload-area">
              <input 
                type="file" 
                accept=".csv,.xlsx,.json"
                onChange={(e) => setFile(e.target.files[0])}
                className="file-input"
                id="fileInput"
              />
              <label htmlFor="fileInput" className="file-label">
                <div className="upload-icon">
                  {file ? (
                    <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : (
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  )}
                </div>
                <div className="upload-text">
                  {file ? (
                    <div>
                      <p className="file-name">{file.name}</p>
                      <p className="file-ready">Ready to analyze</p>
                    </div>
                  ) : (
                    <div>
                      <p className="upload-prompt">Click to select file</p>
                      <p className="upload-hint">CSV, Excel, or JSON files</p>
                    </div>
                  )}
                </div>
              </label>
            </div>
            
            <div className="prompt-section">
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
              className="analyze-button"
            >
              {loading ? (
                <div className="loading-content">
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {currentStep || 'Analyzing...'}
                </div>
              ) : (
                'Analyze Data'
              )}
            </button>
          </motion.div>

          {/* Error Section */}
          {error && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="error-card"
            >
              <div className="error-content">
                <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <h3>Analysis Error</h3>
                  <p>{error}</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Results Section */}
          {results && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="results-container"
            >
              {/* EDA Results */}
              {results.eda?.analysis && (
                <div className="eda-section">
                  <h2>📈 Exploratory Data Analysis</h2>
                  
                  {/* Visualizations */}
                  {results.eda.analysis.visualizations && results.eda.analysis.visualizations.length > 0 && (
                    <div className="visualizations-grid">
                      {results.eda.analysis.visualizations.map((chart, index) => (
                        <motion.div 
                          key={index} 
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.4, delay: index * 0.1 }}
                          className="chart-card"
                        >
                          <h3>{chart.title}</h3>
                          <p className="chart-description">{chart.description}</p>
                          <div className="chart-badge">
                            {chart.category} • {chart.chart_type}
                          </div>
                          
                          {chart.chart_json && (
                            <div className="chart-container">
                              {Plot ? (
                                <Plot
                                  data={JSON.parse(chart.chart_json).data}
                                  layout={{
                                    ...JSON.parse(chart.chart_json).layout,
                                    width: 580,
                                    height: 350,
                                    margin: { l: 50, r: 20, t: 30, b: 40 },
                                    font: { size: 11 }
                                  }}
                                  config={{ displayModeBar: false, responsive: true }}
                                  style={{ width: '100%', height: '100%' }}
                                  onError={(err) => console.error('Plotly error:', err)}
                                />
                              ) : (
                                <div className="chart-loading">
                                  📊 Loading chart...
                                </div>
                              )}
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* EDA Insights */}
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
            </motion.div>
          )}
        </div>
      </motion.main>

      <style jsx>{`
        .analyze-page {
          min-height: 100vh;
          font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: white;
          color: #1f2937;
        }

        .nav-bar {
          display: flex;
          justify-content: center;
          padding: 2rem 0 1rem;
        }

        .nav-container {
          display: flex;
          background: #f3f4f6;
          border-radius: 9999px;
          padding: 0.25rem;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .nav-item {
          padding: 0.5rem 1.5rem;
          border-radius: 9999px;
          font-weight: 500;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.3s ease;
          transform: scale(1);
        }

        .nav-item:hover {
          background: #e5e7eb;
          transform: scale(1.05);
        }

        .nav-active {
          background: #4b5563;
          color: white;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .page-header {
          text-align: center;
          padding: 1rem 2rem 2rem;
        }

        .page-header h1 {
          font-size: 2rem;
          font-weight: bold;
          margin-bottom: 0.5rem;
          color: #1f2937;
        }

        .page-header p {
          color: #6b7280;
          font-size: 1rem;
        }

        .main-content {
          padding: 0 2rem 2rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .content-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .service-status-card, .upload-card {
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 1rem;
          padding: 1.5rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .service-status-card h2, .upload-card h2 {
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 1rem;
          color: #1f2937;
        }

        .check-services-btn {
          background: #3b82f6;
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.3s ease;
          margin-bottom: 1rem;
        }

        .check-services-btn:hover {
          background: #2563eb;
          transform: translateY(-1px);
        }

        .services-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
        }

        .service-item {
          padding: 1rem;
          border-radius: 0.5rem;
          text-align: center;
          transition: all 0.3s ease;
        }

        .service-item.healthy {
          border: 2px solid #10b981;
          background: #ecfdf5;
        }

        .service-item.down {
          border: 2px solid #ef4444;
          background: #fef2f2;
        }

        .status-indicator {
          display: block;
          margin: 0.5rem 0;
          font-weight: 500;
        }

        .error-text {
          color: #ef4444;
          font-size: 0.75rem;
        }

        .file-upload-area {
          margin-bottom: 1.5rem;
        }

        .file-input {
          display: none;
        }

        .file-label {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          padding: 2rem;
          border: 2px dashed #d1d5db;
          border-radius: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
          background: #f9fafb;
        }

        .file-label:hover {
          border-color: #9ca3af;
          background: #f3f4f6;
          transform: scale(1.02);
        }

        .upload-icon {
          flex-shrink: 0;
        }

        .upload-text {
          text-align: center;
        }

        .file-name {
          font-weight: 600;
          color: #10b981;
          margin-bottom: 0.25rem;
        }

        .file-ready {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .upload-prompt {
          font-weight: 500;
          color: #1f2937;
          margin-bottom: 0.25rem;
        }

        .upload-hint {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .prompt-section {
          margin-bottom: 1.5rem;
        }

        .prompt-textarea {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 0.5rem;
          font-family: inherit;
          resize: vertical;
          transition: all 0.3s ease;
        }

        .prompt-textarea:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .analyze-button {
          background: #4b5563;
          color: white;
          border: none;
          padding: 0.75rem 2rem;
          border-radius: 0.5rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.3s ease;
          width: 100%;
          font-size: 1rem;
        }

        .analyze-button:hover:not(:disabled) {
          background: #374151;
          transform: translateY(-1px);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .analyze-button:disabled {
          background: #d1d5db;
          cursor: not-allowed;
          transform: none;
        }

        .loading-content {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .error-card {
          background: #fef2f2;
          border: 1px solid #fecaca;
          border-radius: 1rem;
          padding: 1.5rem;
        }

        .error-content {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .error-content h3 {
          color: #dc2626;
          font-weight: 600;
          margin: 0 0 0.25rem 0;
        }

        .error-content p {
          color: #7f1d1d;
          margin: 0;
        }

        .results-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .eda-section, .ml-section {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 1rem;
          padding: 2rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .eda-section h2, .ml-section h2 {
          font-size: 1.5rem;
          font-weight: 600;
          margin-bottom: 1.5rem;
          color: #1f2937;
        }

        .visualizations-grid, .analyses-container {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .chart-card, .analysis-card {
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 0.75rem;
          padding: 1.5rem;
          box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        }

        .chart-card h3, .analysis-card h3 {
          font-size: 1.125rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
          color: #1f2937;
        }

        .chart-description {
          color: #6b7280;
          font-size: 0.875rem;
          margin-bottom: 0.75rem;
        }

        .chart-badge {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          background: #dbeafe;
          color: #1d4ed8;
          border-radius: 9999px;
          font-size: 0.75rem;
          font-weight: 500;
          margin-bottom: 1rem;
        }

        .chart-container {
          width: 100%;
          height: 350px;
          background: white;
          border-radius: 0.5rem;
          overflow: hidden;
          border: 1px solid #e5e7eb;
        }

        .chart-loading, .chart-error {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: #6b7280;
          font-size: 0.875rem;
        }

        .chart-error {
          background: #fef3c7;
          color: #92400e;
        }

        .insights-section {
          margin-top: 1.5rem;
        }

        .insights-container {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
        }

        .insight-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.75rem;
          padding: 1.5rem;
        }

        .insight-card h3 {
          color: #3b82f6;
          font-size: 1.125rem;
          font-weight: 600;
          margin-bottom: 1rem;
        }

        .insight-card ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .insight-card li {
          padding: 0.5rem 0;
          border-bottom: 1px solid #f3f4f6;
          line-height: 1.5;
          color: #374151;
        }

        .insight-card li:last-child {
          border-bottom: none;
        }

        .no-insights {
          text-align: center;
          padding: 2rem;
          background: #eff6ff;
          border: 1px solid #bfdbfe;
          border-radius: 0.75rem;
          color: #1e40af;
          font-style: italic;
        }

        .tee-attestation-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.75rem;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .tee-attestation-card h3 {
          font-size: 1.125rem;
          font-weight: 600;
          margin-bottom: 1rem;
          color: #1f2937;
        }

        .attestation-content {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
        }

        .status-success {
          color: #10b981;
        }

        .status-failed {
          color: #ef4444;
        }

        .status-success, .status-failed {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
        }

        .status-icon {
          font-size: 1.25rem;
          flex-shrink: 0;
        }

        .status-success strong, .status-failed strong {
          display: block;
          margin-bottom: 0.25rem;
        }

        .status-success p, .status-failed p {
          font-size: 0.875rem;
          margin: 0;
        }

        .metadata-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 0.75rem;
        }

        .metadata-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 0;
          border-bottom: 1px solid #f3f4f6;
        }

        .metadata-item:last-child {
          border-bottom: none;
        }

        .metadata-item label {
          font-weight: 500;
          color: #374151;
          font-size: 0.875rem;
        }

        .metadata-item code {
          background: #f3f4f6;
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-family: 'SF Mono', Monaco, monospace;
          font-size: 0.75rem;
          color: #1f2937;
        }

        .verification-info {
          grid-column: 1 / -1;
          margin-top: 1rem;
          padding: 1rem;
          background: #f9fafb;
          border-radius: 0.5rem;
        }

        .verification-info h4 {
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
          color: #1f2937;
        }

        .verification-info p {
          font-size: 0.875rem;
          line-height: 1.5;
          color: #374151;
          margin: 0;
        }

        .analysis-insights {
          margin-bottom: 1.5rem;
        }

        .analysis-insights h4 {
          color: #10b981;
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }

        .analysis-insights ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .analysis-insights li {
          padding: 0.25rem 0;
          line-height: 1.4;
          color: #374151;
        }

        .analysis-charts {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1.5rem;
        }

        .no-ml-results {
          text-align: center;
          padding: 2rem;
          background: #fef3c7;
          border: 1px solid #fbbf24;
          border-radius: 0.75rem;
          color: #92400e;
        }

        .no-ml-results h4 {
          margin-bottom: 0.5rem;
        }

        .no-ml-results p {
          margin: 0;
        }

        @media (max-width: 768px) {
          .visualizations-grid, .analyses-container {
            grid-template-columns: 1fr;
          }
          
          .attestation-content {
            grid-template-columns: 1fr;
          }
          
          .chart-container {
            height: 300px;
          }
        }
      `}</style>
    </div>
  );
}
